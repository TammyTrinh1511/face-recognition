import cv2
import os
import numpy as np
from tqdm import tqdm
from retinaface import RetinaFace
import skimage.filters as filters


# Find the Euclidean distance between two 2D vectors
def euclidean(u, v):
    return np.sqrt(sum([(i - j) ** 2 for i, j in zip(u, v)]))

def extract_face(img, face_detector, resolution=(100, 100), align=True, scale=0.9, single=True):
    img = img[..., ::-1]
    
    objs = RetinaFace.detect_faces(img, model=face_detector, threshold=0.9)
    results = []

    if isinstance(objs, dict):
        for identity in list(objs.values()):
            landmarks = identity["landmarks"]
            face_img = align_face(img, landmarks, resolution, align, scale)
            face_img = face_img[..., ::-1]
            f = identity["facial_area"]            
            results.append({"face_img": face_img, "coords": {"x": f[0], "y": f[1], "w": f[2] - f[0], "h": f[3] - f[1]}})
            if single:
                break
    else:
        print("Warning: Face could not be detected. Using original image. If this is already a cropped face image, consider setting <extract> to False.")
        results.append({"face_img": img[..., ::-1], "coords": {"x": 0, "y": 0, "w": 0, "h": 0}})
    return results

def detect_faces(input_path, output_path, resolution, align=True, scale=0.9):

    accepted_extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    labels = [fn for fn in os.listdir(input_path) if fn.split(".")[-1] in accepted_extensions]

    for label in tqdm(labels):

        img = cv2.imread(os.path.join(input_path, label))
        # Turns the BGR image to RGB
        img = img[..., ::-1]

        # Detect the faces using RetinaFace
        face_detector = RetinaFace.build_model()
        objs = RetinaFace.detect_faces(img, model=face_detector, threshold=0.9)

        if isinstance(objs, dict):
            for index, identity in enumerate(list(objs.values())):
                facial_area = identity["facial_area"]

                y = facial_area[1]
                h = facial_area[3] - y
                x = facial_area[0]
                w = facial_area[2] - x
                confidence = identity["score"]

                landmarks = identity["landmarks"]
                face_img = align_face(img, landmarks, resolution, align, scale)

                cv2.imwrite(os.path.join(output_path, f"{label.split('.')[0]}_{index}.jpg"), face_img[..., ::-1])

# align the face based on the position of the eyes before extracting the face
def align_face(img, landmarks, resolution, align=True, scale=1):

    # define the width of the left eye to the right eye, and the height of the eyes
    eye_pos_w, eye_pos_h = 0.3 * scale, 0.35 * scale
    width, height = resolution[0], resolution[1]

    # find the location of the eyes and their centre
    l_e = landmarks['left_eye']
    r_e = landmarks['right_eye']
    center = (((r_e[0] + l_e[0]) // 2), ((r_e[1] + l_e[1]) // 2))

    # find the distance between the eyes
    dx = (r_e[0] - l_e[0])
    dy = (r_e[1] - l_e[1])
    dist = euclidean(l_e, r_e)

    # find the angle between the eyes
    angle = np.degrees(np.arctan2(dy, dx)) + 180 if align else 0
    scale = width * (1 - (2 * eye_pos_w)) / dist

    # find the x and y translations needed to center the image
    tx = width * 0.5
    ty = height * eye_pos_h

    # get the affine matrix needed to perform the transformation
    m = cv2.getRotationMatrix2D(center, angle, scale)

    m[0, 2] += (tx - center[0])
    m[1, 2] += (ty - center[1])

    face_align = cv2.warpAffine(img, m, (width, height))

    return face_align

def blur(img):
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    return blurred

# Correct the lighting of an image to remove differences in illumination
def correct_lighting(img):

    # Generate a blurred copy of the original image
    smooth = cv2.GaussianBlur(img, (33,33), 0)

    # Divide the original image by the blurred copy
    division = cv2.divide(img, smooth, scale=255)

    # Apply an unsharp mask to the divided copy
    sharp = filters.unsharp_mask(division, radius=1, amount=0.1, channel_axis=True, preserve_range=False)

    # Clip the pixel brightness from 0 to 255
    sharp = (255*sharp).clip(0,255).astype(np.uint8)
    return sharp

def adjust_img(img, resolution=(100, 100), is_correct_lighting=True, is_blur=True, is_extract_face=True):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_correct_lighting:
        img = correct_lighting(img)
    if is_blur:
        img = blur(img)
    img = cv2.resize(img, resolution)
    return img

def preprocess(db_path, resolution=(100, 100), is_correct_lighting=True, is_blur=True, is_extract_face=True, scale=1, verbose=False):

    file_name = f"/processed.pkl"
    if os.path.exists(db_path + file_name):

        if not verbose:
            print(
                f"WARNING: Processed images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            obj = pickle.load(f)

        if not verbose:
            print("There are ", len(obj["vectors"]), " processed images found in ", file_name)
    
    else:
        img_paths = []
        labels = []
        imgs = []
        
        if is_extract_face:
            face_detector = RetinaFace.build_model()

        for r, _, f in tqdm(os.walk(db_path)):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "\\" + file

                    img = cv2.imread(exact_path)
                    if is_extract_face:
                        output = extract_face(img, face_detector=face_detector, resolution=resolution, scale=scale)
                        max_index = np.argmax([output[i]["coords"]["w"] for i in range(len(output))])
                        value = output[max_index]
                        img = value["face_img"]
                        cv2.imwrite(exact_path, img)
                    
                    img = adjust_img(
                        img = img,
                        resolution=resolution,
                        is_correct_lighting=is_correct_lighting,
                        is_blur=is_blur,
                        is_extract_face=is_extract_face
                    )

                    imgs.append(img)
                    img_paths.append(exact_path)
                    labels.append(r.split("\\")[-1])

        if len(img_paths) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )
        
        labels = np.array(labels)
        vectors = np.array([img.flatten() for img in imgs])
        
        obj = {
            "img_paths": img_paths,
            "labels": labels,
            "imgs": imgs,
            "vectors": vectors
            }

        save_data(obj, os.path.join(db_path + file_name))

    return obj

def preprocess(db_path, resolution=(100, 100), is_correct_lighting=True, is_blur=True, is_extract_face=True, scale=1, verbose=False):

    file_name = f"/processed.pkl"
    if os.path.exists(db_path + file_name):

        if not verbose:
            print(
                f"WARNING: Processed images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            obj = pickle.load(f)

        if not verbose:
            print("There are ", len(obj["vectors"]), " processed images found in ", file_name)
    
    else:
        img_paths = []
        labels = []
        imgs = []
        
        if is_extract_face:
            face_detector = RetinaFace.build_model()

        for r, _, f in tqdm(os.walk(db_path)):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "\\" + file

                    img = cv2.imread(exact_path)
                    if is_extract_face:
                        output = extract_face(img, face_detector=face_detector, resolution=resolution, scale=scale)
                        max_index = np.argmax([output[i]["coords"]["w"] for i in range(len(output))])
                        value = output[max_index]
                        img = value["face_img"]
                        cv2.imwrite(exact_path, img)
                    
                    img = adjust_img(
                        img = img, 
                        resolution=resolution,
                        is_correct_lighting=is_correct_lighting,
                        is_blur=is_blur,
                        is_extract_face=is_extract_face
                    )

                    imgs.append(img)
                    img_paths.append(exact_path)
                    labels.append(r.split("\\")[-1])

        if len(img_paths) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )
        
        labels = np.array(labels)
        vectors = np.array([img.flatten() for img in imgs])
        
        obj = {
            "img_paths": img_paths,
            "labels": labels,
            "imgs": imgs,
            "vectors": vectors
            }

        save_data(obj, os.path.join(db_path + file_name))

    return obj


x = preprocess(r'custom_data/', resolution=(100, 100), is_correct_lighting=True, is_blur=True, is_extract_face=True, scale=1, verbose=False)
print(x)