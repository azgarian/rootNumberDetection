
# Adım adım yaptıklarımız

env: aiproject

- Hasta tomografi ve panoramik görüntüleri seçilip hard diske aktarıldı.
- `patient_info.csv` ve `patient_to_id.csv` isimli iki dosyada hastaların bilgileri düzenlendi.
  - patient_info kolonları:
    - patient_id: hastaya atanan id numarası
    - age: yaş bilgisi
    - gender: cinsiyet bilgisi
    - radiology: (True/False) True ise tomografi net değil radyolojiye sorulacak
    - 15_root_num: 15 numaralı dişin kök sayısı
    - 25_root_num: 25 numaralı dişin kök sayısı
    - 15_exact_img: Tomografiden çekilecek fotograf karesi aralığı 15 numaralı diş için
    - 25_exact_img: Tomografiden çekilecek fotograf karesi aralığı 25 numaralı diş için
    - 15_crop_coordinate: Seçilen karelerden sadece 15 numaralı dişi kesmek için kullanılan koordinatlar
    - 25_crop_coordinate: Seçilen karelerden sadece 25 numaralı dişi kesmek için kullanılan koordinatlar
    - 15_pano_crop_coordinate: 15 numaralı dişi kesmek için kullanılan koordinatlar (Panoramik görüntü)
    - 25_pano_crop_coordinate: 25 numaralı dişi kesmek için kullanılan koordinatlar (Panoramik görüntü)
  - patient_to_id kolonları:
    - patient_name: hastanın isim bilgisi
    - patient_id: hastaya atanan id numarası
    - file_name: hastanın hard disk'teki dosya adı
- Dicom formatındaki hasta tomografilerinden kullanılacak kareler seçildi. (`image_tomo.ipynb` ve `get_image.py`)
- `define_crop_coord.ipynb` kodunu kullanarak 15 25 numaralı dişlerin kordinatları belirlendi.
- Seçilen kareler hedef dişleri (15 ve 25 numara) içerecek şekilde kırpıldı. (`crop_image.ipynb`)
- Panoramik görüntüleri de işlemek için `pano` klasörü içinde düzenlendi. (`pano_rename.ipynb`)
- Panoramik görüntülerin kesilecek koordinatları belirlendi. (`crop_pano_image.ipynb`)

## Yapılacaklar

- machine learning bak.
