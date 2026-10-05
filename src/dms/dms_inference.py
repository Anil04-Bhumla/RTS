import os
import cv2
import numpy as np

class DMSInfer:
    def __init__(self, model, in_shape=(160, 160), device='cuda:0'):
        # type: (str, tuple, str) -> None
        """
        :param model: experiment name, a corresponding 'exp_name_sim.onnx' must exist in ./weights
        :param in_shape: image input shape, tuple
        """

        self.device = device
        self.detection_size = (in_shape, in_shape)

        model_file = os.path.join(os.path.dirname(__file__), "../../models/", model)
        ext = model_file.split('.')[-1]

        if ext == "onnx":
            
            import onnxruntime as ort

            ep = "CUDAExecutionProvider" if "cuda" in device else "CPUExecutionProvider"
            print(f"[NOTE]: Inference using OnnxRuntime with {ep}")

            self.sess = ort.InferenceSession(model_file, providers=[ep])
            self.dd = len(self.sess.get_outputs()) == 6
            self.input_name = self.sess.get_inputs()[0].name

            # normalization
            mean = np.array([103.53, 116.28, 123.675])
            std = np.array([58.82, 58.82, 58.82])

            self.mean = np.reshape(mean, (3,1,1))
            self.std = np.reshape(std, (3,1,1))
            self.detect_internal = self._detect_onnx

        else:
            raise ValueError(f"Invalid model {ext}")

    def crop_image(self, orig, bbox):

        bbox = bbox.copy()
        image = orig.copy()
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        face_width = (1 + 2 * 0.25) * bbox_width
        face_height = (1 + 2 * 0.05) * bbox_height
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 1.8]
        bbox[0] = max(0, center[0] - face_width // 2)
        bbox[1] = max(0, center[1] - face_height // 2)
        bbox[2] = min(image.shape[1], center[0] + face_width // 2)
        bbox[3] = min(image.shape[0], center[1] + face_height // 2)
        bbox = bbox.astype(int)
        crop_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] #y1:y2, x1:x2
        h, w, _ = crop_image.shape

        if h <= 0 or w <= 0:
            return np.zeros((160, 160, 3), dtype=np.uint8), ([160, 160, bbox[1], bbox[0]])

        crop_image = cv2.resize(crop_image, (160, 160))
        return crop_image, ([h, w, bbox[1], bbox[0]])

    def _detect_onnx(self, crop_image):

        # Pre processing
        crop_image = crop_image.transpose(2,0,1) # hwc->chw
        crop_image = (crop_image - self.mean) / self.std
        crop_image = np.expand_dims(crop_image, axis=0)
        crop_image = crop_image.astype(np.float32)
        raw = self.sess.run(None, {self.input_name: crop_image})

        return raw

    def detect(self, img, bbox,):

        crop_image, detail = self.crop_image(img, bbox)
        raw = self.detect_internal(crop_image)

        landmark = raw[0].reshape((-1, 2))
        euler, eyes, mouth_s = raw[1][0], raw[2][0], raw[3][0]
        euler = euler * 90
        mouth_s = np.argmax(mouth_s)
        #act, act_score = raw[4][0], raw[5][0]
        act = raw[4][0]
        act = np.argmax(act)

        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3]
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2]

        return landmark, euler, eyes, mouth_s, act
