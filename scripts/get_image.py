import os
import matplotlib.pyplot as plt
import matplotlib
import pydicom
matplotlib.use('Agg')

def process_patient(patient_id, patient_name2id, patient_target, dicom_dir):
    patient_dir = os.path.join("data","tomo", "images", patient_id)
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)

    if len(os.listdir(patient_dir)) != 100:

        patient = patient_name2id[patient_name2id["patient_id"] == patient_id]["file_name"].iloc[0]
        dicom_file = os.path.join(dicom_dir, f"{patient}.dcm")
        curr_dicom = pydicom.dcmread(dicom_file)

        coords = (
            patient_target.loc[
                            patient_target["patient_id"] == patient_id, "15_exact_img"
                            ].str.split("-") +
            patient_target.loc[
                            patient_target["patient_id"] == patient_id, "25_exact_img"
                            ].str.split("-")
        ).tolist()[0]
        midpoint = (int(max(coords)) + int(min(coords))) // 2
        upper_limit, lower_limit = midpoint + 50, midpoint - 49

        fig_interval = 1
        num = int((upper_limit - lower_limit + 1) / fig_interval)

        fig, ax = plt.subplots()

        for i in range(num):
            ax.axis('off')
            ax.imshow(curr_dicom.pixel_array[lower_limit + i * fig_interval], cmap=plt.cm.bone)
            plt.savefig(os.path.join(patient_dir, f"{lower_limit + i * fig_interval}.jpeg"),
                        bbox_inches='tight', pad_inches=0)
            ax.clear()

        plt.close(fig)
