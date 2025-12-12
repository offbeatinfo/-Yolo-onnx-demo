import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Union
import json
from ultralytics import YOLO

class YOLOv11Inference:
    def __init__(self, model_path: str, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.names = None
        self.colors = None
        
    def load_model(self):
        """Load the YOLOv11 PyTorch model"""
        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.model.conf = self.conf_threshold
        self.model.iou = self.iou_threshold
        self.names = self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print("Model loaded successfully!")
    
    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def run_inference(self, img: np.ndarray) -> dict:
        """Run inference on input image"""
        if self.model is None:
            self.load_model()
        
        # Run inference
        results = self.model(img)
        
        # FIXED: Parse results using new Ultralytics API
        result = results[0]  # results is a list, get first element
        
        if len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)
            pred = np.concatenate([xyxy, conf, cls], axis=1)
        else:
            pred = np.empty((0, 6))
        
        # Convert to list of detections
        detections = []
        for *xyxy, conf, cls in pred:
            detections.append({
                'bbox': [float(coord) for coord in xyxy],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': self.names[int(cls)]
            })
        
        return {
            'detections': detections,
            'image_shape': img.shape[:2]
        }
    
    def draw_detections(self, img: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detections on the image"""
        img_copy = img.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[det['class_id'] % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Draw label text
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img_copy


class ONNXPredictor:
    def __init__(self, onnx_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.onnx_path = onnx_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = (640, 640)  # YOLO11 default input size
        
    def load_model(self):
        """Load the ONNX model"""
        print(f"Loading ONNX model from {self.onnx_path}...")
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self.input_shape = (input_shape[2], input_shape[3])
        
        print(f"ONNX model loaded successfully!")
        print(f"Input name: {self.input_name}")
        print(f"Input shape: {input_shape}")
        print(f"Output names: {self.output_names}")
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize and pad image while maintaining aspect ratio"""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, r, (dw, dh)
    
    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int]]:
        """Preprocess image for ONNX model"""
        # Store original shape
        original_shape = img.shape[:2]
        
        # Apply letterbox (same as PyTorch preprocessing)
        img_letterbox, ratio, (dw, dh) = self.letterbox(img, self.input_shape)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        
        # Convert HWC to CHW and add batch dimension
        img_input = img_rgb.transpose(2, 0, 1)  # HWC to CHW
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)  # Add batch dimension
        img_input = img_input / 255.0  # Normalize to [0, 1]
        
        return img_input, ratio, (dw, dh), original_shape
    
    def xywh2xyxy(self, x):
        """Convert boxes from [x_center, y_center, w, h] to [x1, y1, x2, y2]"""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
        return y
    
    def nms(self, boxes, scores, iou_threshold):
        """Apply Non-Maximum Suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def run_inference(self, img: np.ndarray, names: dict = None) -> dict:
        """Run inference using ONNX model"""
        if self.session is None:
            self.load_model()
        
        # Preprocess image
        img_input, ratio, (dw, dh), original_shape = self.preprocess_image(img)
        
        print(f"Preprocessed image shape: {img_input.shape}")
        
        # Run inference - use only first output
        outputs = self.session.run([self.output_names[0]], {self.input_name: img_input})
        output = outputs[0]
        
        print(f"ONNX output shape: {output.shape}")
        
        # Process output based on shape
        # YOLO11 output is typically (1, 84, 8400) where:
        # - 1 is batch size
        # - 84 is [4 bbox coords + 80 class scores]
        # - 8400 is number of predictions
        
        if len(output.shape) == 3:
            batch_size, num_features, num_predictions = output.shape
            
            # Transpose to (batch_size, num_predictions, num_features)
            if num_features < num_predictions:
                output = output.transpose(0, 2, 1)
                num_features, num_predictions = num_predictions, num_features
            
            print(f"Reshaped output: {output.shape}")
            
            # Process detections
            detections = []
            
            for batch_idx in range(output.shape[0]):
                predictions = output[batch_idx]  # Shape: (num_predictions, num_features)
                
                # Extract boxes and scores
                boxes = predictions[:, :4]  # First 4 values are bbox coords
                class_scores = predictions[:, 4:]  # Remaining are class scores
                
                # Get max class score and class id for each prediction
                max_scores = np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)
                
                # Filter by confidence threshold
                mask = max_scores > self.conf_threshold
                boxes = boxes[mask]
                scores = max_scores[mask]
                class_ids = class_ids[mask]
                
                print(f"Detections after confidence threshold: {len(boxes)}")
                
                if len(boxes) == 0:
                    continue
                
                # Convert from xywh to xyxy
                boxes = self.xywh2xyxy(boxes)
                
                # Apply NMS for each class
                final_boxes = []
                final_scores = []
                final_class_ids = []
                
                for class_id in np.unique(class_ids):
                    class_mask = class_ids == class_id
                    class_boxes = boxes[class_mask]
                    class_scores = scores[class_mask]
                    
                    # Apply NMS
                    keep_indices = self.nms(class_boxes, class_scores, self.iou_threshold)
                    
                    final_boxes.extend(class_boxes[keep_indices])
                    final_scores.extend(class_scores[keep_indices])
                    final_class_ids.extend([class_id] * len(keep_indices))
                
                # Convert boxes back to original image coordinates
                for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
                    # Adjust for padding
                    x1 = (box[0] - dw) / ratio
                    y1 = (box[1] - dh) / ratio
                    x2 = (box[2] - dw) / ratio
                    y2 = (box[3] - dh) / ratio
                    
                    # Clip to image bounds
                    x1 = max(0, min(x1, original_shape[1]))
                    y1 = max(0, min(y1, original_shape[0]))
                    x2 = max(0, min(x2, original_shape[1]))
                    y2 = max(0, min(y2, original_shape[0]))
                    
                    class_name = names[int(class_id)] if names else f'class_{int(class_id)}'
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(score),
                        'class_id': int(class_id),
                        'class_name': class_name
                    })
        
        print(f"Final ONNX detections: {len(detections)}")
        
        return {
            'detections': detections,
            'image_shape': img.shape[:2]
        }


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate areas
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return iou


def compare_detections(pytorch_detections, onnx_detections, iou_threshold=0.5):
    """Compare detections between PyTorch and ONNX models"""
    matches = []
    unmatched_pytorch = []
    unmatched_onnx = []
    
    # Create copies to track which detections have been matched
    pytorch_dets = pytorch_detections.copy()
    onnx_dets = onnx_detections.copy()
    
    # Match detections based on IoU
    for i, pt_det in enumerate(pytorch_dets):
        if pt_det is None:
            continue
            
        best_iou = iou_threshold
        best_match = None
        
        for j, onnx_det in enumerate(onnx_dets):
            if onnx_det is None:
                continue
                
            iou = calculate_iou(pt_det['bbox'], onnx_det['bbox'])
            
            if iou > best_iou and pt_det['class_id'] == onnx_det['class_id']:
                best_iou = iou
                best_match = j
        
        if best_match is not None:
            matches.append({
                'pytorch': pt_det,
                'onnx': onnx_dets[best_match],
                'iou': best_iou
            })
            # Mark as matched
            pytorch_dets[i] = None
            onnx_dets[best_match] = None
        else:
            unmatched_pytorch.append(pt_det)
    
    # Add remaining ONNX detections as unmatched
    unmatched_onnx = [d for d in onnx_dets if d is not None]
    
    return {
        'matches': matches,
        'unmatched_pytorch': unmatched_pytorch,
        'unmatched_onnx': unmatched_onnx
    }


def save_results(results: dict, output_dir: str = 'output'):
    """Save detection results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON
    with open(os.path.join(output_dir, 'detections.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {os.path.abspath(output_dir)}/")


def main():
    # Initialize paths
    model_path = 'yolo11n.pt'
    image_path = 'image.png'
    onnx_path = 'yolo11n.onnx'
    output_dir = 'output'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process image
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # 1. Run PyTorch inference
    print("\n=== Running PyTorch Inference ===")
    pytorch_predictor = YOLOv11Inference(model_path, image_path)
    pytorch_results = pytorch_predictor.run_inference(image.copy())
    
    # Draw and save PyTorch detections
    pytorch_image = pytorch_predictor.draw_detections(
        image.copy(), pytorch_results['detections']
    )
    cv2.imwrite(os.path.join(output_dir, 'pytorch_detections.jpg'), pytorch_image)
    
    print(f"PyTorch detections: {len(pytorch_results['detections'])}")
    for det in pytorch_results['detections']:
        print(f"  - {det['class_name']}: {det['confidence']:.3f}")
    
    # 2. Convert to ONNX
    print("\n=== Converting to ONNX ===")
    if not os.path.exists(onnx_path):
        print(f"Converting {model_path} to ONNX format...")
        # Use Ultralytics export method
        if pytorch_predictor.model is None:
            pytorch_predictor.load_model()
        
        # Export using Ultralytics method (much easier!)
        pytorch_predictor.model.export(format='onnx', simplify=True)
        print(f"Model converted to ONNX format: {onnx_path}")
    else:
        print(f"ONNX model already exists at {onnx_path}")
    
    # 3. Run ONNX inference
    print("\n=== Running ONNX Inference ===")
    onnx_predictor = ONNXPredictor(onnx_path)
    onnx_results = onnx_predictor.run_inference(image.copy(), names=pytorch_predictor.names)
    
    # Draw and save ONNX detections
    onnx_image = image.copy()
    for det in onnx_results['detections']:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(onnx_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(onnx_image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(output_dir, 'onnx_detections.jpg'), onnx_image)
    print(f"ONNX detections: {len(onnx_results['detections'])}")
    for det in onnx_results['detections']:
        print(f"  - {det['class_name']}: {det['confidence']:.3f}")
    
    # 4. Compare detections (Optional)
    print("\n=== Comparing Detections ===")
    comparison = compare_detections(
        pytorch_results['detections'],
        onnx_results['detections']
    )
    
    print(f"Matching detections: {len(comparison['matches'])}")
    print(f"PyTorch-only detections: {len(comparison['unmatched_pytorch'])}")
    print(f"ONNX-only detections: {len(comparison['unmatched_onnx'])}")
    
    # Print IoU for matches
    if comparison['matches']:
        print("\nMatching detections with IoU:")
        for match in comparison['matches']:
            print(f"  - {match['pytorch']['class_name']}: IoU = {match['iou']:.3f}")
    
    # Save comparison results
    comparison_results = {
        'matches': [
            {
                'pytorch': m['pytorch'],
                'onnx': m['onnx'],
                'iou': float(m['iou'])
            }
            for m in comparison['matches']
        ],
        'unmatched_pytorch': comparison['unmatched_pytorch'],
        'unmatched_onnx': comparison['unmatched_onnx']
    }
    
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # 5. Create a side-by-side visualization
    combined_image = np.hstack((pytorch_image, onnx_image))
    cv2.imwrite(os.path.join(output_dir, 'comparison.jpg'), combined_image)
    
    print("\n=== Results Summary ===")
    print(f"- PyTorch detections: {len(pytorch_results['detections'])}")
    print(f"- ONNX detections: {len(onnx_results['detections'])}")
    print(f"- Matching detections (IoU > 0.5): {len(comparison['matches'])}")
    print(f"- PyTorch-only detections: {len(comparison['unmatched_pytorch'])}")
    print(f"- ONNX-only detections: {len(comparison['unmatched_onnx'])}")
    
    # Calculate average IoU
    if comparison['matches']:
        avg_iou = np.mean([m['iou'] for m in comparison['matches']])
        print(f"- Average IoU for matches: {avg_iou:.3f}")
    
    print(f"\nResults and visualizations saved to '{output_dir}/' directory:")
    print(f"  - pytorch_detections.jpg")
    print(f"  - onnx_detections.jpg")
    print(f"  - comparison.jpg (side-by-side)")
    print(f"  - comparison_results.json")


if __name__ == "__main__":
    main()