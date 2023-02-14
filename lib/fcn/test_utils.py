import sys
import os

from fcn.test_dataset import filter_labels_depth, crop_rois, clustering_features, match_label_crop


sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'MSMFormer'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from tqdm import trange
import torch
import cv2
from fcn.test_common import _vis_minibatch_segmentation, _vis_features, _vis_minibatch_segmentation_final
from utils.evaluation import multilabel_metrics
from detectron2.engine import DefaultPredictor

# ignore some warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from nms import nms


# Reference: https://www.reddit.com/r/computervision/comments/jb6b18/get_binary_mask_image_from_detectron2/

def get_confident_instances(outputs, topk=False, score=0.7, num_class=2, low_threshold=0.4):
    """
    Extract objects with high prediction scores.
    """
    instances = outputs["instances"]
    if topk:
        # we need to remove background predictions
        # keep only object class
        if num_class >= 2:
            instances = instances[instances.pred_classes == 1]
            confident_instances = instances[instances.scores > low_threshold]
            return confident_instances
        else:
            return instances
    confident_instances = instances[instances.scores > score]
    return confident_instances
    

def combine_masks_with_NMS(instances):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
    """
    mask = instances.get('pred_masks').to('cpu').numpy()
    scores = instances.get('scores').to('cpu').numpy()

    # non-maximum suppression
    keep = nms(mask, scores, thresh=0.7).astype(int)
    mask = mask[keep]
    scores = scores[keep]

    num, h, w = mask.shape
    bin_mask = np.zeros((h, w))
    score_mask = np.zeros((h, w))
    num_instance = len(mask)
    bbox = np.zeros((num_instance, 5), dtype=np.float32)

    # if there is not any instance, just return a mask full of 0s.
    if num_instance == 0:
        return bin_mask, score_mask, bbox

    for m, object_label in zip(mask, range(2, 2 + num_instance)):
        label_pos = np.nonzero(m)
        bin_mask[label_pos] = object_label
        score_mask[label_pos] = int(scores[object_label - 2] * 100)

        # bounding box
        y1 = np.min(label_pos[0])
        y2 = np.max(label_pos[0])
        x1 = np.min(label_pos[1])
        x2 = np.max(label_pos[1])
        bbox[object_label - 2, :] = [x1, y1, x2, y2, scores[object_label - 2]]

    return bin_mask, score_mask, bbox

def combine_masks(instances):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
    """
    mask = instances.get('pred_masks').to('cpu').numpy()
    num, h, w = mask.shape
    bin_mask = np.zeros((h, w))
    num_instance = len(mask)
    # if there is not any instance, just return a mask full of 0s.
    if num_instance == 0:
        return bin_mask

    for m, object_label in zip(mask, range(2, 2+num_instance)):
        label_pos = np.nonzero(m)
        bin_mask[label_pos] = object_label
    # filename = './bin_masks/001.png'
    # cv2.imwrite(filename, bin_mask)
    return bin_mask

class Predictor_RGBD(DefaultPredictor):

    def __call__(self, sample):
        """
        Args:
            sample: a dict of a data sample
            # ignore: original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            height, width = 480, 640
            original_image = cv2.imread(sample["file_name"])
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            transforms = self.aug.get_transform(original_image)
            image = transforms.apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            #image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}

            if self.cfg.INPUT.INPUT_IMAGE == "DEPTH" or "RGBD" in self.cfg.INPUT.INPUT_IMAGE:
                depth_image = sample["raw_depth"]
                depth_image = transforms.apply_image(depth_image)
                depth_image = torch.as_tensor(depth_image.astype("float32").transpose(2, 0, 1))
                depth = depth_image
                inputs["depth"] = depth

            predictions = self.model([inputs])[0]
            return predictions

class Network_RGBD(DefaultPredictor):

    def __call__(self, sample):
        """
        Args:
            sample: a dict of a data sample
            # ignore: original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            predictions = self.model([sample])[0]
            return predictions
            
            
def test_sample(cfg, sample, predictor, visualization = False, topk=False, confident_score=0.9, low_threshold=0.4):
    im = cv2.imread(sample["file_name"])
    print(sample["file_name"])
    if "label" in sample.keys():
        gt = sample["label"].squeeze().numpy()
    else:
        gt = sample["labels"].squeeze().numpy()

    image = sample['image_color'].cuda()
    sample["image"] = image
    sample["height"] = image.shape[-2]  # image: 3XHXW, tensor
    sample["width"] = image.shape[-1]
    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    # binary_mask = combine_masks(confident_instances)
    binary_mask, score_mask, bbox = combine_masks_with_NMS(confident_instances)
    metrics = multilabel_metrics(binary_mask, gt)
    print(f"metrics: ", metrics)
    ## Visualize the result
    if visualization:
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]
        # cv2.imwrite(sample["file_name"][-6:-3]+"pred.png", visual_result)
        cv2.imshow("image", visual_result)
        cv2.waitKey(0)
        # cv2.waitKey(100000)
        cv2.destroyAllWindows()
    # markers = refine_with_watershed(im, binary_mask)
    # metrics2 = multilabel_metrics(markers, gt)
    # print(f"metrics2: ", metrics2g)
    return metrics


def get_result_from_network(cfg, image, depth, label, predictor, topk=False, confident_score=0.7, low_threshold=0.4, vis_crop=False):
    height = image.shape[-2]  # image: 3XHXW, tensor
    width = image.shape[-1]
    image = torch.squeeze(image, dim=0)
    if depth is not None:
        depth = torch.squeeze(depth, dim=0)

    sample = {"image": image, "height": height, "width": width, "depth": depth}
    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    if vis_crop:
        im = image.cpu().numpy().transpose((1, 2, 0)) * 255.0
        im += np.array([[[102.9801, 115.9465, 122.7717]]])
        im = im.astype(np.uint8)
        cv2.imshow("image", im)
        cv2.waitKey(0)
        depth_blob = depth.cpu().numpy()
        depth = depth_blob[2]
        plt.imshow(depth)
        plt.axis('off')
        plt.show()

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]
        cv2.imshow("image_segmentation", visual_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    binary_mask, score_mask, bbox = combine_masks_with_NMS(confident_instances)
    return binary_mask


def test_sample_crop(cfg, sample, predictor, predictor_crop, visualization = False, topk=False, confident_score=0.7, low_threshold=0.4, print_result=False):
    # First network: the image needs the original one.
    image = sample['image_color'].cuda() # for future crop
    sample["image"] = image
    sample["height"] = image.shape[-2] # image: 3XHXW, tensor
    sample["width"] = image.shape[-1]
    gt = None
    if "label" in sample.keys():
        gt = sample["label"].squeeze().numpy()
    elif "label" in sample.keys():
        gt = sample["labels"].squeeze().numpy()

    if gt is not None:
        label = torch.from_numpy(gt).unsqueeze(dim=0).cuda()
    if cfg.MODEL.USE_DEPTH:
        depth = sample['depth'].cuda()
    else:
        depth = None

    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    binary_mask, score_mask, bbox = combine_masks_with_NMS(confident_instances)
    metrics = multilabel_metrics(binary_mask, gt)
    if print_result:
        print("file name: ", sample["file_name"])
        print("first:", metrics)

    if visualization:
        im = cv2.imread(sample["file_name"])  # this is for visualization
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]
        # cv2.imwrite(sample["file_name"][-6:-3]+"pred.png", visual_result)
        cv2.imshow("image", visual_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    out_label = torch.as_tensor(binary_mask).unsqueeze(dim=0).cuda()
    if depth is not None:
        if len(depth.shape) == 3:
            depth = torch.unsqueeze(depth, dim=0)

    if len(image.shape) == 3:
        image = torch.unsqueeze(image, dim=0)
    if depth is not None:
        # filter labels on zero depth
        if 'OSD' in sample["file_name"]:
            out_label = filter_labels_depth(out_label, depth, 0.8)
        else:
            out_label = filter_labels_depth(out_label, depth, 0.5)

    # zoom in refinement
    out_label_refined = None
    if predictor_crop is not None:
        rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        if rgb_crop.shape[0] > 0:
            labels_crop = torch.zeros((rgb_crop.shape[0], rgb_crop.shape[-2], rgb_crop.shape[-1]))#.cuda()
            for i in range(rgb_crop.shape[0]):
                if depth_crop is None:
                    binary_mask_crop = get_result_from_network(cfg, rgb_crop[i], None, out_label_crop[i],
                                                               predictor_crop,
                                                               topk=topk, confident_score=confident_score,
                                                               low_threshold=low_threshold)
                else:
                    binary_mask_crop = get_result_from_network(cfg, rgb_crop[i], depth_crop[i], out_label_crop[i], predictor_crop,
                                                       topk=topk, confident_score=confident_score, low_threshold=low_threshold)
                labels_crop[i] = torch.from_numpy(binary_mask_crop)
            out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

        if visualization and rgb_crop.shape[0] > 0:
            bbox = None
            _vis_minibatch_segmentation_final(image, depth, label, out_label, out_label_refined, None,
                selected_pixels=None, bbox=bbox)

    if out_label_refined is not None:
        out_label_refined = out_label_refined.squeeze(dim=0).cpu().numpy()
    prediction = out_label.squeeze().detach().cpu().numpy()
    if out_label_refined is not None:
        prediction_refined = out_label_refined
    else:
        prediction_refined = prediction.copy()
    metrics_refined = multilabel_metrics(prediction_refined, gt)
    if print_result:
        print("refined: ", metrics_refined)
        print("========")

    return metrics, metrics_refined


def test_sample_crop_nolabel(cfg, sample, predictor, predictor_crop, visualization = False, topk=False, confident_score=0.7, low_threshold=0.4, print_result=False):
    image = sample['image_color'].cuda() # for future crop
    sample["image"] = image
    if len(image.shape) == 4:
        image = torch.squeeze(image, dim=0)
        print("image shape: ", image.shape)
    sample["height"] = image.shape[-2] # image: 3XHXW, tensor
    sample["width"] = image.shape[-1]

    if cfg.MODEL.USE_DEPTH:
        depth = sample['depth'].cuda()
        if len(depth.shape) == 4:
            depth = torch.squeeze(depth, dim=0)
    else:
        depth = None
        sample['depth'] = None

    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    binary_mask, score_mask, bbox = combine_masks_with_NMS(confident_instances)

    from detectron2.utils.visualizer import Visualizer, GenericMask, ColorMode, _create_text_labels
    from detectron2.utils.colormap import random_color
    from collections import Counter
    import matplotlib.colors as mplc
    import matplotlib as mpl
    import numpy as np

    from sklearn.utils import shuffle
    from sklearn.cluster import KMeans

    class MyVisualizer(Visualizer):

        def draw_instance_predictions(self, predictions):
            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
            labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
            keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

            if predictions.has("pred_masks"):
                masks = np.asarray(predictions.pred_masks)
                masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
            else:
                masks = None

            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    [x / 255 for x in self.metadata.thing_colors[c]] for c in classes
                ]
                alpha = 0.8
            else:
                colors = None
                alpha = 0.9

            self.overlay_instances(
                masks=masks,
                boxes=boxes,
                labels=labels,
                keypoints=keypoints,
                assigned_colors=colors,
                alpha=alpha,
            )
            return self.output

        def overlay_instances(
                self,
                *,
                boxes=None,
                labels=None,
                masks=None,
                keypoints=None,
                assigned_colors=None,
                alpha=0.5,
        ):
            num_instances = 0
            if boxes is not None:
                boxes = self._convert_boxes(boxes)
                num_instances = len(boxes)
            if masks is not None:
                masks = self._convert_masks(masks)
                if num_instances:
                    assert len(masks) == num_instances
                else:
                    num_instances = len(masks)
            if keypoints is not None:
                if num_instances:
                    assert len(keypoints) == num_instances
                else:
                    num_instances = len(keypoints)
                keypoints = self._convert_keypoints(keypoints)
            if labels is not None:
                assert len(labels) == num_instances
            if assigned_colors is None:
                assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
            if num_instances == 0:
                return self.output
            if boxes is not None and boxes.shape[1] == 5:
                return self.overlay_rotated_instances(
                    boxes=boxes, labels=labels, assigned_colors=assigned_colors
                )

            # Display in largest to smallest order to reduce occlusion.
            areas = None
            if boxes is not None:
                areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            elif masks is not None:
                areas = np.asarray([x.area() for x in masks])

            if areas is not None:
                sorted_idxs = np.argsort(-areas).tolist()
                masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
                assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
                keypoints = keypoints[sorted_idxs] if keypoints is not None else None

            for i in range(num_instances):
                color = assigned_colors[i]
                if masks is not None:
                    for segment in masks[i].polygons:
                        self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            return self.output

        def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
            color = mplc.to_rgb(color) + (1,)
            polygon = mpl.patches.Polygon(
                segment,
                fill=True,
                facecolor=color,
                edgecolor=color,
                linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
            )
            self.output.ax.add_patch(polygon)
            return self.output

    if visualization:
        im = cv2.imread(sample["file_name"])  # this is for visualization
        im = np.zeros_like(im)
        v = MyVisualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]

        masked_image = np.copy(visual_result)

        w, h, d = masked_image.shape
        image_data = np.reshape(masked_image, (w * h, d))

        color_dict = {}
        for i in range(w*h):
            if str(image_data[i]) in color_dict:
                color_dict[str(image_data[i])] += color_dict[str(image_data[i])] + 1
            else:
                color_dict[str(image_data[i])] = 1

        import heapq
        from operator import itemgetter

        top_colors =  sorted(color_dict, key=color_dict.get, reverse=True)[:8]

        for i in range(w*h):
            if str(image_data[i]) not in top_colors:
                image_data[i] = np.array([0, 0, 0])

        visual_result = image_data.reshape((w, h, -1))

        # SEED = 0
        # train_data = shuffle(image_data, random_state=SEED, n_samples=100)
        # print(train_data.shape)
        # kmeans = KMeans(n_clusters=8, random_state=SEED)
        # kmeans = kmeans.fit(train_data)
        #
        # pred = kmeans.predict(image_data)
        # final_image = kmeans.cluster_centers_[pred].reshape(w, h, -1)
        #
        # visual_result = final_image



        cv2.imwrite(sample["file_name"][-6:-3]+"pred.png", visual_result)
        # cv2.imshow("image", visual_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    out_label = torch.as_tensor(binary_mask).unsqueeze(dim=0).cuda()
    out_score = torch.as_tensor(score_mask).unsqueeze(dim=0).cuda()    
    if depth is not None:
        if len(depth.shape) == 3:
            depth = torch.unsqueeze(depth, dim=0)
    if len(image.shape) == 3:
        image = torch.unsqueeze(image, dim=0)
    if depth is not None:
        # filter labels on zero depth
        if 'file_name' in sample and 'OSD' in sample["file_name"]:
            out_label = filter_labels_depth(out_label, depth, 0.8)
        else:
            out_label = filter_labels_depth(out_label, depth, 0.5)

    # zoom in refinement
    out_label_refined = None
    if predictor_crop is not None:
        rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        if rgb_crop.shape[0] > 0:
            labels_crop = torch.zeros((rgb_crop.shape[0], rgb_crop.shape[-2], rgb_crop.shape[-1]))#.cuda()
            for i in range(rgb_crop.shape[0]):
                if depth is not None:
                    binary_mask_crop = get_result_from_network(cfg, rgb_crop[i], depth_crop[i], out_label_crop[i], predictor_crop,
                                                           topk=topk, confident_score=confident_score, low_threshold=low_threshold)
                else:
                    binary_mask_crop = get_result_from_network(cfg, rgb_crop[i], None, out_label_crop[i],
                                                               predictor_crop,
                                                               topk=topk, confident_score=confident_score,
                                                               low_threshold=low_threshold)
                labels_crop[i] = torch.from_numpy(binary_mask_crop)
            out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

    if visualization:
        bbox = None
        _vis_minibatch_segmentation_final(image, depth, None, out_label, out_label_refined, None,
            selected_pixels=None, bbox=bbox)

    if out_label_refined is not None:
        out_label_refined = out_label_refined.squeeze(dim=0).cpu().numpy()
    prediction = out_label.squeeze().detach().cpu().numpy()
    if out_label_refined is not None:
        prediction_refined = out_label_refined
    else:
        prediction_refined = prediction.copy()
        
    return out_label, out_label_refined, out_score, bbox        
        

def test_dataset(cfg,dataset, predictor, visualization=False, topk=False, confident_score=0.7, low_threshold=0.4):
    metrics_all = []
    for i in trange(len(dataset)):
        metrics = test_sample(cfg, dataset[i], predictor, visualization=visualization,
                              topk=topk, confident_score=confident_score, low_threshold=low_threshold)
        metrics_all.append(metrics)

    print('========================================================')
    if not topk:
        print("Mask threshold: ", confident_score)
    else:
        print(f"We first pick top {cfg.TEST.DETECTIONS_PER_IMAGE} instances ")
        print(f"and get those whose class confidence > {low_threshold}.")

    print("weight: ", cfg.MODEL.WEIGHTS)
    result = {}
    num = len(metrics_all)
    print('%d images' % num)
    print('========================================================')
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]

    for k in sorted(result.keys()):
        result[k] /= num
        print('%s: %f' % (k, result[k]))

    print('%.6f' % (result['Objects Precision']))
    print('%.6f' % (result['Objects Recall']))
    print('%.6f' % (result['Objects F-measure']))
    print('%.6f' % (result['Boundary Precision']))
    print('%.6f' % (result['Boundary Recall']))
    print('%.6f' % (result['Boundary F-measure']))
    print('%.6f' % (result['obj_detected_075_percentage']))

    print('========================================================')
    print(result)
    print('====================END=================================')

def test_dataset_crop(cfg,dataset, predictor, network_crop, visualization=False, topk=True, confident_score=0.9, low_threshold=0.4):
    metrics_all = []
    metrics_all_refined = []
    for i in trange(len(dataset)):
        metrics, metrics_refined = test_sample_crop(cfg, dataset[i], predictor, network_crop, visualization=visualization,
                              topk=topk, confident_score=confident_score, low_threshold=low_threshold)
        metrics_all.append(metrics)
        metrics_all_refined.append(metrics_refined)
    print('========================================================')
    if not topk:
        print("Mask threshold: ", confident_score)
    else:
        print(f"We first pick top {cfg.TEST.DETECTIONS_PER_IMAGE} instances ")
        print(f"and get those whose class confidence > {low_threshold}.")

    print("weight: ", cfg.MODEL.WEIGHTS)
    result = {}
    num = len(metrics_all)
    print('%d images' % num)
    print('========================================================')
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]

    for k in sorted(result.keys()):
        result[k] /= num
        print('%s: %f' % (k, result[k]))

    print('%.6f' % (result['Objects Precision']))
    print('%.6f' % (result['Objects Recall']))
    print('%.6f' % (result['Objects F-measure']))
    print('%.6f' % (result['Boundary Precision']))
    print('%.6f' % (result['Boundary Recall']))
    print('%.6f' % (result['Boundary F-measure']))
    print('%.6f' % (result['obj_detected_075_percentage']))

    print('========================================================')
    print(result)
    print('====================Refined=============================')

    result_refined = {}
    for metrics in metrics_all_refined:
        for k in metrics.keys():
            result_refined[k] = result_refined.get(k, 0) + metrics[k]

    for k in sorted(result_refined.keys()):
        result_refined[k] /= num
        print('%s: %f' % (k, result_refined[k]))
    print(result_refined)
    print('========================================================')




