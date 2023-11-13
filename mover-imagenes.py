import os
import shutil

photos_male_path = 'person-face-sketches/test/photos-male'
photos_female_path = 'person-face-sketches/test/photos-female'
sketches_path = 'person-face-sketches/test/sketches'
sketches_male_path = 'person-face-sketches/test/sketches-male'
sketches_female_path = 'person-face-sketches/test/sketches-female'

os.makedirs(sketches_male_path, exist_ok=True)
os.makedirs(sketches_female_path, exist_ok=True)

def move_sketches(photos_path, sketches_dest_path):
    for photo in os.listdir(photos_path):
        photo_id = os.path.splitext(photo)[0]
        sketch_file = photo_id + '.jpg'

        sketch_path = os.path.join(sketches_path, sketch_file)
        if os.path.exists(sketch_path):
            shutil.move(sketch_path, os.path.join(sketches_dest_path, sketch_file))

move_sketches(photos_male_path, sketches_male_path)
move_sketches(photos_female_path, sketches_female_path)
