import os
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
from skimage.util import invert

""" Code of Augmentations
    GB: Gaussian Blur
    DL: Dilation
    ER: Erosion
    WT: Watermark
    RR: Resolution Reduction
    SPN: Salt & Pepper Noise
    MB: Motion Blur
    RBC: Random Black Conversion
    RT: Random Rotation
    RP: Random Padding
"""
class GaussianBlur:
    def __init__(self, kernel_size=5):
        self.kernel_size = (kernel_size, kernel_size)
    def __call__(self, img):
        return cv2.GaussianBlur(img, self.kernel_size, 0)
    def __repr__(self):
        return "GB_(K={0})_".format(self.kernel_size)
        
class Dilation:
    def __init__(self, iterations=1):
        self.iterations = iterations
        # Apply binary thresholding (if not already binary)
    def __call__(self, img):
        self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)# np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)# np.random.randint(0, 2, size=(self.kernel_size, self.kernel_size), dtype=np.uint8) / 2
        # y = cv2.dilate(img, self.kernel, iterations=self.iterations)
        # _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # inverted_image = invert(binary_image)
        # skeleton = morph.skeletonize(inverted_image)
        # skeleton = invert(skeleton)
        # z = np.minimum(y, skeleton)
        # t = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        # t_rgb = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        results = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.kernel)
        return results
        
    def __repr__(self):
        return "DL_"

class Erosion:
    def __init__(self, kernel_size, iterations=1):
        self.kernel_size = kernel_size
        self.iterations = iterations
    def __call__(self, img):
        self.kernel = np.random.randint(0, 2, size=(self.kernel_size, self.kernel_size), dtype=np.uint8)
        return cv2.erode(img, self.kernel, iterations=self.iterations)
        
    def __repr__(self):
        return "ER(K=" + str(self.kernel) + ")_"

class Watermark:
    def __init__(self, text, color=(255, 0, 0), watermark_angle=10, opacity=0.3):
        self.watermark_text = text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 3
        self.thickness = 5
        self.angle = watermark_angle
        self.opacity = opacity
    def __call__(self, img):
        self.opacity = random.random() / 5
        angle = random.uniform(-self.angle, self.angle)
        self.color = random.choice([(255, 0, 0), (127, 127, 127), (0, 0, 255), (0, 255, 0), (0, 255, 255)])
        overlay = np.ones_like(img, dtype=np.uint8) * 255
        (text_width, text_height), _ = cv2.getTextSize(self.watermark_text, self.font, self.font_scale, self.thickness)
        x = (overlay.shape[1] - text_width) // 2
        y = (overlay.shape[0] + text_height) // 2
        cv2.putText(overlay, self.watermark_text, (x, y), self.font, self.font_scale, self.color, self.thickness, cv2.LINE_AA)
        center = (overlay.shape[1] // 2, overlay.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_overlay = cv2.warpAffine(overlay, rotation_matrix, (overlay.shape[1], overlay.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        watermarked_image = cv2.addWeighted(rotated_overlay, self.opacity, img, 1 - self.opacity, 0)
        return watermarked_image
    def __repr__(self):
        return "WT_(color=" + str(self.color) + ")_"

class Pixelation:
    def __init__(self, reduction_ratio=2):
        self.upper_ratio = reduction_ratio
    def __call__(self, img):
        self.reduction_ratio = (random.uniform(1.1, self.upper_ratio), random.uniform(1.1, self.upper_ratio))
        height, width =  img.shape[0], img.shape[1]
        small_h = int(height / self.reduction_ratio[0])
        small_w = int(width / self.reduction_ratio[1])
        small_image = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)
        return pixelated_image
    def __repr__(self):
        return "RR_(R=" + str(self.reduction_ratio) + ")_"

class Noise:
    def __init__(self, salt_prob, pepper_prob):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
    def __call__(self, img):
        noisy_image = img.copy()
        num_salt = np.ceil(self.salt_prob * img.size).astype(int)
        num_pepper = np.ceil(self.pepper_prob * img.size).astype(int)
    
        # Add Salt
        coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
        noisy_image[coords[0], coords[1]] = 200
    
        # Add Pepper
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
        noisy_image[coords[0], coords[1]] = 100
    
        return noisy_image
    def __repr__(self):
        return "SPN_"

class MotionBlur:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
    def __call__(self, img):
        kernel_motion_blur = np.zeros((self.kernel_size, self.kernel_size))
        kernel_motion_blur[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
        kernel_motion_blur = kernel_motion_blur / self.kernel_size
        motion_blur_image = cv2.filter2D(img, -1, kernel_motion_blur)
        return motion_blur_image
    def __repr__(self):
        return "MB_"

class RandomOcclusion:
    def __init__(self, color=(255, 0, 0), opacity=0.1):
        self.color = random.choice([(0, 0, 255), (0, 255, 0), (0, 255, 255)])
        
    def __call__(self, image):
        self.opacity = random.random() / 5
        self.color = random.choice([(0, 0, 255), (0, 255, 0), (0, 255, 255)])
        overlay = image.copy()
        blend_image = image.copy()
        # Add random rectangles with RGBA colors
        num_rectangles = 2
        for _ in range(num_rectangles):
            x1 = np.random.randint(0, image.shape[1] // 3)
            y1 = np.random.randint(0, image.shape[0] // 3)
            x2 = np.random.randint(image.shape[1] // 2, image.shape[1])
            y2 = np.random.randint(image.shape[0] // 2, image.shape[0])
        
            # Draw rectangle on the overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color, -1)
        
            # Blend the rectangle with the original image based on alpha
            blend_image = cv2.addWeighted(overlay, self.opacity, image, 1 - self.opacity, 0)
        return blend_image
    def __repr__(self):
        return "RO_(C=" + str(self.color) + ")_"

class RandomBlackConversion:
    def __init__(self, ratio=0.2):
        self.upper = ratio
    def __call__(self, image):
        self.ratio = random.uniform(0.005, self.upper)
        binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = (binary_image > 127).astype(np.uint8)
        black_points = np.argwhere(binary_image == 0)
        num_points_to_convert = int(len(black_points) * self.ratio)
        # Randomly select black points to convert
        if len(black_points) > 0:  # Ensure there are black points to choose from
            selected_indices = np.random.choice(len(black_points), 
                                                size=min(num_points_to_convert, len(black_points) // 2), 
                                                replace=False)
            selected_points = black_points[selected_indices]
        
            # Convert the selected black points to white (value 1)
            for point in selected_points:
                binary_image[point[0], point[1]] = 1
        # Save the modified image or display it
        modified_image = (binary_image * 255).astype(np.uint8)
        output_image = cv2.cvtColor(modified_image, cv2.COLOR_GRAY2RGB)
        return output_image
    def __repr__(self):
        return "RBC_(R=" + str(self.ratio) + ")_"

class RandomRotation:
    def __init__(self, angle=2):
        self.upper = angle
    def __call__(self, image):
        # Get the image dimensions
        self.angle = random.uniform(-self.upper, self.upper)
        (height, width) = image.shape[:2]
        
        # Define the center of the image
        center_x = random.randint(width // 4, 3 * width // 4)
        center_y = random.randint(height // 4, 3 * height // 4)
        self.center = (center_x, center_y)
        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(self.center, self.angle, 1.0)
        # Apply the rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated_image
    def __repr__(self):
        return "RT_(C=" + str(self.center) + ",A=" + str(self.angle) + ")_"

class RandomPadding:
    def __init__(self, upper_paddings):
        if isinstance(upper_paddings, int):
            a = upper_paddings
            upper_paddings = [a, a, a, a]
        elif isinstance(upper_paddings, list):
            pass
        else:
            print("[WARNING] 'upper_paddings' must be an integer or list of integers")
        self.upper_paddings = upper_paddings
    def __call__(self, image):
        self.text_paddings = (random.randint( self.upper_paddings[0] // 2, self.upper_paddings[0]),
                              random.randint( self.upper_paddings[1] // 2, self.upper_paddings[1]),
                              random.randint( self.upper_paddings[2] // 2, self.upper_paddings[2]),
                              random.randint( self.upper_paddings[3] // 2, self.upper_paddings[3]))
        original_height, original_width = image.shape[:2]
        padded_image = cv2.copyMakeBorder(
            image, 
            self.text_paddings[0], self.text_paddings[1], self.text_paddings[2], self.text_paddings[3], 
            borderType=cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)
        )
        # Resize the padded image back to the original size
        resized_image = cv2.resize(padded_image, (original_width, original_height))
        return resized_image
    def __repr__(self):
        return "RP_(P=" + str(self.text_paddings) + ")_"
        
def generate_augmentation():
    augment_groups = [
        [RandomBlackConversion(ratio=0.01),  Dilation(kernel_size=3)],
        [Erosion(kernel_size=3)],
        [MotionBlur(kernel_size=3)],
        [Pixelation(reduction_ratio=2)],
        [Watermark(text="Document", opacity=0.2, color=(127, 127, 127), watermark_angle=10), RandomOcclusion()],
        [RandomRotation(angle=3)],
        [RandomPadding(upper_paddings=10)]
    ]
    selection = []
    for i, group in enumerate(augment_groups):
        if random.random() < 0.6:
            if len(group) == 1:
                if random.random() < 0.7:
                    selection.append(group[0])
            else:
                if random.random() < 0.5:
                    selection.append(group[0])
                else:
                    selection.append(group[1])
    if len(selection) > 0:
        fixed_element = selection[0]
        remaining_elements = selection[1:]
        random.shuffle(remaining_elements)
        result = [fixed_element] + remaining_elements
        return result
    else:
        return []
def generate_augmentation_detection(aug_ratio = 0.4):
    augment_groups = [
        [Dilation()],
        [Erosion(kernel_size=3)],
        [MotionBlur(kernel_size=5)],
        [Pixelation(reduction_ratio=4)],
    ]
    selection = []
    for i, group in enumerate(augment_groups):
        if random.random() < aug_ratio:
            selection.append(group[0])
            
    if len(selection) > 1:
        fixed_element = selection[0]
        remaining_elements = selection[1:]
        random.shuffle(remaining_elements)
        result = [fixed_element] + remaining_elements
        return result
    else:
        return []

def eval_augmentation():
    augment_groups = [
        [Pixelation(reduction_ratio=2)],
        [Watermark(text="Document", opacity=0.2, color=(127, 127, 127), watermark_angle=10), RandomOcclusion()],
        [RandomRotation(angle=3)],
        [RandomPadding(upper_paddings=10)]
    ]
    selection = []
    for i, group in enumerate(augment_groups):
        if len(group) == 1:
            if random.random() < 0.7:
                selection.append(group[0])
        else:
            if random.random() < 0.5:
                selection.append(group[0])
            else:
                selection.append(group[1])
    fixed_element = selection[0]
    remaining_elements = selection[1:]
    random.shuffle(remaining_elements)
    result = [fixed_element] + remaining_elements
    return result



class HKTAugment(object):
    def __init__(self):
        print("Initialize HKT Augmentation...")
    def __call__(self, img):
        self.augments = generate_augmentation()
        temp_img = img.copy()
        name_augments = ""
        for augment in self.augments:
            temp_img = augment(temp_img)
            name_augments += str(augment)
        return temp_img, name_augments

class HKTAugmentDetection(object):
    def __init__(self):
        print("Initialize HKT Augmentation...")
    def __call__(self, img):
        self.augments = generate_augmentation_detection()
        temp_img = img.copy()
        name_augments = ""
        for augment in self.augments:
            temp_img = augment(temp_img)
            name_augments += str(augment)
        return temp_img, name_augments

class EvalHKTAugment(object):
    def __init__(self):
        print("Initialize Evaluation HKT Augmentation...")
    def __call__(self, img):
        self.augments = eval_augmentation()
        temp_img = img.copy()
        for augment in self.augments:
            temp_img = augment(temp_img)
        return temp_img