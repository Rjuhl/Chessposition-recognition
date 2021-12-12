from PIL import Image
import json
from tqdm import tqdm


classes_to_dir = {
 'black-knight': 'BK',
 'white-bishop': 'WB',
 'black-pawn': 'BP',
 'black-bishop': 'BB',
 'white-queen': 'WQ',
 'black-queen': 'BQ',
 'black-rook': 'BR',
 'white-knight': 'WK',
 'black-king': 'BKi',
 'white-pawn': 'WP',
 'white-king': 'WKi',
 'white-rook': 'WR',
 'bishop': "BB",
}

read_dir = "../data/robo"
write_dir = "../data/final-data"

out_width, out_height = 224, 224


def extract_images(directory):
    full_path = f"{read_dir}/{directory}"
    with open(f"{full_path}/_annotations.createml.json") as f:
        annotations = json.load(f)
    for image in tqdm(annotations):
        uid = 0
        fn = image["image"]
        for label in image['annotations']:
            coord = label["coordinates"]
            bbox = (coord["x"], coord["y"], coord["width"], coord["height"])
            img = modify_image(f"{full_path}/{fn}", out_width, out_height, crop=bbox)
            res_dir = classes_to_dir[label['label']]
            img.save(f"{write_dir}/{res_dir}/{fn}-{uid}.jpg", "JPEG")
            uid += 1


def modify_image(img_path, ds_w, ds_h, crop=None):
    img = Image.open(img_path)
    if crop is not None:
        x, y, w, h = crop
        img = img.crop((x - w/2, y - h/2, x + w/2, y + h/2))
    else:
        w, h = img.size
    if w / h < ds_w / ds_h:
        # Height is limiting factor, pad width
        new_height = ds_h
        new_width = round(w * ds_h / h)
    else:
        new_width = ds_w
        new_height = round(h * ds_w / w)
    result = Image.new(img.mode, (ds_w, ds_h), (0, 0, 0))
    result.paste(img.resize((new_width, new_height), Image.ANTIALIAS),
                 (round(ds_w / 2 - new_width / 2), round(ds_h / 2 - new_height / 2)))
    return result


if __name__ == "__main__":
    print("Extracting train")
    extract_images("train")
    print("Extracting valid")
    extract_images("valid")
