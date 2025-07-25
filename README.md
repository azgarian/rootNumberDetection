# README

## Workflow Order (for rerun)

env: aiproject

scripts:

- [`image_tomo.ipynb`](src/preprocess/image_tomo.ipynb)
- [`define_crop_coord.ipynb`](src/preprocess/define_crop_coord.ipynb) # if a redefine needed.
- [`crop_image.ipynb`](src/preprocess/crop_image.ipynb)
- [`pano_rename.ipynb`](src/preprocess/pano/pano_rename.ipynb)
- [`crop_pano_image.ipynb`](src/preprocess/crop_pano_image.ipynb)

## Step-by-Step Actions Taken

- Selected patient tomography and panoramic images were transferred to the hard disk.
- Patient information was organized in two files named [`patient_info.csv`](data/patient_info.csv) and [`patient_to_id.csv`](data/patient_to_id.csv).
  - Columns in `patient_info`:
    - `patient_id`: ID number assigned to the patient
    - `age`: age information
    - `gender`: gender information
    - `radiology`: (True/False) If True, the tomography is unclear and will be referred to radiology
    - `15_root_num`: number of roots for tooth number 15
    - `25_root_num`: number of roots for tooth number 25
    - `15_exact_img`: frame range to be extracted from tomography for tooth number 15
    - `25_exact_img`: frame range to be extracted from tomography for tooth number 25
    - `15_crop_coordinate`: coordinates used to crop only tooth number 15 from the selected frames
    - `25_crop_coordinate`: coordinates used to crop only tooth number 25 from the selected frames
    - `15_pano_crop_coordinate`: coordinates used to crop tooth number 15 (panoramic image)
    - `25_pano_crop_coordinate`: coordinates used to crop tooth number 25 (panoramic image)
  - Columns in `patient_to_id`:
    - `patient_name`: name information of the patient
    - `patient_id`: ID number assigned to the patient
    - `file_name`: file name of the patient on the hard disk
- Frames to be used from patient tomographies in Dicom format were selected. ([`image_tomo.ipynb`](src/preprocess/image_tomo.ipynb) and [`get_image.py`](src/preprocess/get_image.py))
- Coordinates for teeth numbered 15 and 25 were determined using the [`define_crop_coord.ipynb`](src/preprocess/define_crop_coord.ipynb) code.
- Selected frames were cropped to include the target teeth (numbers 15 and 25). ([`crop_image.ipynb`](src/preprocess/crop_image.ipynb))
- Panoramic images were also processed and organized in the `pano` folder. ([`pano_rename.ipynb`](src/preprocess/pano_rename.ipynb))
- Coordinates for cropping the panoramic images were determined and cropped. ([`crop_pano_image.ipynb`](src/preprocess/crop_pano_image.ipynb))
- Splitted data into train, validation, and test sets. Balanced classes in train set using augmentation. ([`augmentation.ipynb`](src/preprocess/augmentation.ipynb))


## To Do
