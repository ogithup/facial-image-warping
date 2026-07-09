# Facial Image Warping Projesi
## DSP Dersi ve Bilgisayar Mühendisligi Perspektifiyle Detayli Proje Raporu

**Proje adi:** Facial Image Warping, Aging, and Expression Transformation  
**Rapor tipi:** Ders/proje teknik raporu  
**Hazirlanan kapsam:** Kod tabani, testler, GUI akislari ve kaydedilmis output artefact'lari kullanilarak

---

## 1. Projenin Amaci

Bu projede bir yuz goruntusu, klasik anlamda sadece bir fotograf olarak degil, **islenebilir bir sayisal sinyal** olarak ele alindi. Ana hedef, kullanicidan alinan goruntuyu alip asama asama:

1. dogrulamak,
2. standardize etmek,
3. yuz bolgesini ayirmak,
4. bu bolgeyi landmark noktalarla temsil etmek,
5. geometri tabanli donusumler uygulamak,
6. frekans uzayinda analiz etmek,
7. sonuclari nicel metriklerle degerlendirmek,
8. bunlari GUI ve real-time webcam uzerinden sunmakti.

Bu acidan proje, klasik bilgisayar gorusu, goruntu isleme, sayisal isaret isleme ve yazilim muhendisligini ayni cati altinda birlestiren moduler bir sistem haline geldi.

---

## 2. Bu Projede Neler Ogrendim

Bu proje kapsami icinde DSP dersi ve bilgisayar muhendisligi mantigiyla ogrendigim temel noktalar sunlardir:

### 2.1 Goruntunun bir sinyal oldugu fikri

Bir goruntu, iki boyutlu ayrik bir sinyaldir. Bu nedenle:

- uzamsal alanda piksel yogunluklariyla,
- frekans alaninda spektral bilesenlerle,
- geometri alaninda koordinat ve mesh yapilariyla,

calisilabilir.

### 2.2 Pipeline tasarimi

Ham veriyi dogrudan islemek yerine onu asamalara ayirmak gerekir:

`Input -> Preprocessing -> ROI -> Landmark -> Warp -> Evaluation -> Visualization`

Bu ayrim sayesinde sistem hem test edilebilir hale gelir hem de her modulu bagimsiz gelistirmek mumkun olur.

### 2.3 ROI ve bilgi yogunlugu kavrami

Tum resmi islemek yerine yalnizca yuz bolgesini secmek:

- hesap maliyetini azaltir,
- hata kaynaklarini daraltir,
- sonraki asamalarin kararliligini arttirir.

### 2.4 Landmark tabanli temsil

Bir yuzu ham piksel yiginindan ziyade **anlamsal noktalara ayrilmis bir geometri** olarak temsil etmek, warping ve reenactment gibi islemlerin temelidir.

### 2.5 Geometrik warping mantigi

Bir ifadeyi degistirmenin ana yolu, tum resmi filtrelemek degil; yuz mesh'inin belirli bolgelerini hedef koordinatlara tasimaktir.

### 2.6 Fourier ve spektral enerji mantigi

Donusumun yalnizca gorunur sonucuna degil, frekans bilesenlerine de bakmak gerekir. Boylece:

- yuksek frekansli detaylar,
- dusuk frekansli genel aydinlik yapisi,
- goruntu sertligi / yumusakligi

nicel olarak incelenebilir.

### 2.7 Nicel kalite olcumu

Bir donusumun iyi gorunmesi yetmez; `MSE`, `PSNR`, `SSIM` gibi metriklerle degerlendirmek gerekir.

### 2.8 GUI ve real-time sistem farki

Offline image studio ile real-time webcam akis ayni sey degildir. Real-time sistemlerde:

- latency,
- frame skipping,
- landmark smoothing,
- dusuk cozumurlukte analiz,
- render yuku

ayri problem alanlari olusturur.

### 2.9 Klasik warp ile modern expression transfer farki

Klasik yaklasim yuz geometrisini dogrudan deforme eder. Daha gelismis yaklasim ise expression'ı identity'den ayirmaya calisir. Bu projede her iki yonun de temelini kurmus oldum.

---

## 3. Projenin Yuksek Seviyeli Mimarisi

Kod tabaninda merkez orkestra dosyasi `app.py`'dir. Burada butun pipeline asamalari tek bir API gibi toplanir.

Temel giris fonksiyonlari:

- `run_preprocessing_pipeline(...)`
- `run_face_detection_pipeline(...)`
- `run_landmark_pipeline(...)`
- `run_expression_warp_pipeline(...)`
- `run_frequency_analysis_pipeline(...)`
- `run_aging_pipeline(...)`
- `run_deaging_pipeline(...)`
- `run_analysis_pipeline(...)`
- `run_reference_expression_transfer_pipeline(...)`
- `run_realtime_frame_pipeline(...)`

Bu tasarim muhendislik acisindan onemlidir; cunku GUI katmani dogrudan algoritma yazmaz, yalnizca bu servis fonksiyonlarini cagirir.

---

## 4. Kullandigim Ana Dosyalar ve Gorevleri

### 4.1 `src/facial_image_warping/input_module.py`

Bu modul kullanicidan gelen resmi alir, dosya tipini denetler ve OpenCV ile bellekte bir image payload haline getirir.

Burada ogrenilen muhendislik noktasi:

- girdi dogrulama
- hata yonetimi
- metadata tutma
- Windows Unicode path sorunlarini asmak icin `np.fromfile + cv2.imdecode` kullanimi

Kod mantigi:

```python
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def load_image(image_source: str | Path) -> dict:
    image_buffer = np.fromfile(image_path, dtype=np.uint8)
    bgr_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
```

Bu noktada proje bana, goruntu isleme projelerinde "dosya okuma" kisminin bile gercek bir sistem problemi oldugunu gostermis oldu.

### 4.2 `src/facial_image_warping/preprocessing.py`

Bu asama, girdiyi ortak bir formata getirir:

- `BGR -> RGB`
- grayscale
- resize
- normalize
- histogram uretimi

Kod mantigi:

```python
def preprocess_image(image: dict, target_size: tuple[int, int] = (512, 512), save_outputs: bool = True) -> dict:
    rgb_image = convert_bgr_to_rgb(image)
    resized_rgb = resize_to_standard(rgb_image, target_size=target_size)
    grayscale = convert_to_grayscale(resized_rgb)
    normalized = normalize_pixel_values(grayscale)
```

Burada ogrendigim sey, bir DSP sisteminde **normalizasyonun opsiyonel degil zorunlu** oldugudur. Farkli boyut ve farkli renk uzayindaki goruntuleri dogrudan sonraki asamalara vermek kararsizlik uretir.

### 4.3 `src/facial_image_warping/face_detection.py`

Bu modul OpenCV Haar Cascade ile yuzu bulur, en buyuk bounding box'i secer ve ROI crop uretir.

Kod mantigi:

```python
def detect_face_region(image: dict, scale_factor: float = 1.1, min_neighbors: int = 5, target_size: tuple[int, int] = (512, 512), save_outputs: bool = True) -> dict:
    grayscale = convert_to_detection_grayscale(image)
    classifier = load_haar_cascade()
    detections = classifier.detectMultiScale(grayscale, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    bounding_box = select_largest_face(detections)
    cropped_face = crop_face_region(image, bounding_box)
    analysis_face = resize_face_crop(cropped_face, target_size=target_size)
```

Muhendislik yorumu:

- Haar Cascade modern derin ogrenme detector'leri kadar guclu degildir.
- Buna ragmen hizli, bagimsiz ve kurulum maliyeti dusuk oldugu icin pipeline'in ilk surumu icin uygun bir secimdir.
- Bu secim ders kapsaminda "basitten calisan bir sistem kurma" prensibini dogrulamaktadir.

### 4.4 `src/facial_image_warping/landmark_detection.py`

Bu modul MediaPipe Face Mesh kullanarak yuzu 468 landmark noktasiyla temsil eder.

Kod mantigi:

```python
def normalized_landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    x = min(max(int(round(landmark.x * (width - 1))), 0), width - 1)
    y = min(max(int(round(landmark.y * (height - 1))), 0), height - 1)
    return x, y
```

```python
FACE_REGIONS = {
    "eyes": [...],
    "eyebrows": [...],
    "nose": [...],
    "lips": [...],
    "jawline": [...],
    "cheeks": [...],
}
```

Bu kisim bana sunu ogretti:

- model normalized koordinat dondurur,
- ama isleme yapmak icin bu degerleri piksel uzayina cevirmek gerekir,
- her landmark esit onemde degildir,
- anlamsal region haritalari kurmak gerekir.

Bu, ham model cikisini muhendislikte kullanilabilir bir veri yapisina cevirmektir.

### 4.5 `src/facial_image_warping/geometric_warping.py`

Bu dosya geometric expression editing katmanidir.

Burada:

- hedef landmarklar uretildi,
- Delaunay triangulation kuruldu,
- her ucgen affine donusum ile warp edildi,
- sonra butun parcalar birlestirildi.

Kod mantigi:

```python
def apply_delaunay_triangulation(image_shape: tuple[int, ...], landmarks: list[dict]) -> list[tuple[int, int, int]]:
    subdiv = cv2.Subdiv2D((0, 0, width, height))
```

```python
def warp_triangle(source_image, destination_image, source_triangle, target_triangle) -> None:
    warp_matrix = cv2.getAffineTransform(np.float32(src_rect_points), np.float32(dst_rect_points))
    warped_patch = cv2.warpAffine(...)
```

```python
def apply_expression_warp(face_image, landmarks, transformation="smile_enhancement", intensity=0.5, save_outputs=True) -> dict:
    target_landmarks = create_target_landmarks(landmarks, transformation=transformation, intensity=intensity)
    triangles = apply_delaunay_triangulation(source_pixels.shape, landmarks)
```

Burada ogrendigim temel DSP/CG karisimi kavramlar:

- mesh tabanli deformasyon
- affine donusum
- interpolasyon
- lokal geometri koruma
- piksel yeniden esleme

### 4.6 `src/facial_image_warping/fourier_analysis.py`

Bu proje dogrudan DSP dersiyle en guclu sekilde bu dosyada baglanti kuruyor.

Kod mantigi:

```python
def compute_fft(image: dict) -> np.ndarray:
    grayscale = _to_grayscale_float(image)
    fft_result = np.fft.fft2(grayscale)
    return np.fft.fftshift(fft_result)
```

```python
def calculate_frequency_energy(fft_result: np.ndarray) -> dict:
    power_spectrum = np.abs(fft_result) ** 2
    ...
    low_frequency_energy = float(power_spectrum[mask].sum())
    high_frequency_energy = float(power_spectrum[~mask].sum())
```

Bu asama bana sunlari kazandirdi:

- 2D FFT uygulamasi
- frekansin merkeze alinmasi
- magnitude spectrum
- low/high frequency ayirimi
- spektral enerji yorumlama

### 4.7 `src/facial_image_warping/aging_filter.py`

Bu modul aging ve de-aging icin frekans ilhamli filtreler kullanir.

Kod mantigi:

```python
def _high_pass_texture(image: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.subtract(image, blurred)
```

```python
def _local_contrast_boost(image: np.ndarray, clip_limit: float) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
```

```python
def apply_deaging_filter(image: dict, intensity: float = 0.5) -> dict:
    smoothed = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bilateral = cv2.bilateralFilter(...)
```

Muhendislik mantigi:

- aging icin yuksek frekansli mikro detaylar ve lokal kontrast arttirildi,
- de-aging icin dusuk geciren yumusatma ve kenar koruma kullanildi.

Bu, sinyal alaninda "hangi bileseni arttirirsam nasil bir goruntu algisi olusur?" sorusunun pratik cevabidir.

### 4.8 `src/facial_image_warping/expression_transfer.py`

Bu dosya projenin en gelismis bolumlerinden biridir. Burada artik tek bir yuzu filtrelemek yerine, **bir referans yuz ifadesini kaynak yuze tasima** hedeflenmistir.

Desteklenen yontemler:

```python
TRANSFER_METHODS = {
    "safe_classical",
    "tps",
    "expression_coefficients",
}
```

#### a. Safe Classical

- landmark farklarini alir
- secilen bolgelerde target landmark uretir
- triangle veya kontrollu warp mantigina yakin, daha guvenli bir deformasyon yapar

#### b. TPS - Thin Plate Spline

```python
def _warp_image_with_tps(source_pixels, source_landmarks, target_landmarks, regions):
    source_points, target_points = _build_tps_control_points(...)
    weights, affine = _fit_tps_model(target_points, source_points)
    mapped_points = _evaluate_tps_mapping(query_points, target_points, weights, affine)
    return cv2.remap(...)
```

TPS'nin mantigi, mesh ucgenlerini bagimsiz degistirmek yerine daha yumusak bir global deformasyon yuzeyi olusturmaktir.

#### c. Expression Coefficients

```python
def create_expression_coefficient_targets(source_landmarks, reference_landmarks, blend_factor=0.7) -> list[dict]:
    source_coeffs = extract_expression_coefficients(source_landmarks)
    reference_coeffs = extract_expression_coefficients(reference_landmarks)
```

Bu yontem en modern dusunceye yakindir; cunku tum landmark kumesini kopyalamak yerine:

- agiz acikligi
- agiz genisligi
- goz acikligi
- kas yuksekligi

gibi kompakt ifade parametrelerini source'a tasir.

#### d. Ağız ici texture transferi

```python
def _blend_mouth_interior_from_reference(...):
    if reference_coeffs["mouth_open"] < 0.045:
        return base_pixels
```

Bu kisim ozellikle onemlidir; cunku sadece agzi asagi yukari cekmek yetmez. Referans yuzde agiz aciksa, dis ve agiz ici dokusunu da kontrollu sekilde aktarmak gerekir.

Bu bolum bana, gercekci reenactment'in yalnizca geometri degil, **bolgesel goruntu birlestirme** problemi de oldugunu ogretti.

### 4.9 `src/facial_image_warping/evaluation.py`

Bu dosya, donusumlerin yalnizca gorsel degil, sayisal olarak da analiz edilmesini saglar.

Kod mantigi:

```python
def compute_mse(original_image: dict, transformed_image: dict) -> float:
    difference = original_rgb.astype(np.float32) - transformed_rgb.astype(np.float32)
    return float(np.mean(difference ** 2))
```

```python
def compute_psnr(original_image: dict, transformed_image: dict) -> float:
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))
```

```python
def compute_ssim(original_image: dict, transformed_image: dict) -> float:
    return float(structural_similarity(...))
```

Burada ogrenilen nokta:

- MSE hatayi ortalama enerji farki olarak gosterir
- PSNR sinyal/gurultu oranina benzer kalite bakisi sunar
- SSIM insan algisina daha yakin yapi benzerligini olcer

Ek olarak `Absolute Difference` gorseli de uretilmistir.

### 4.10 `app.py`

Bu dosya proje icindeki ana orkestratordur.

Onemli katkilar:

- GUI ile cekirdek algoritmalar arasindaki ayrim
- realtime ile offline akislarin ayri tutulmasi
- webcam icin landmark smoothing
- landmark inference oncesi kucuk crop'lari buyutme

Kod mantigi:

```python
def _prepare_landmark_inference_face(face_image: dict, max_dimension: int = 512) -> tuple[dict, float]:
    if current_max >= max_dimension:
        return face_image, 1.0
```

```python
def _smooth_landmarks(current_landmarks, previous_landmarks, smoothing_alpha: float = 0.65):
    ...
```

```python
def run_realtime_frame_pipeline(frame: np.ndarray, transformation: str = "aging", ...):
    face_result = detect_face_region(frame_image, save_outputs=False)
    landmarks_result = _prepare_landmarks_if_needed(...)
```

Bu, gercek zamanli sistemlerde "tek frame algoritmasi" ile "stream pipeline'i" arasindaki farki anlamami sagladi.

### 4.11 `streamlit_app.py` ve `gui/dashboard.py`

Bu iki dosya, projeyi kullaniciya acan GUI katmanlarini olusturur.

Burada:

- image upload
- webcam capture
- transfer method secimi
- quality profile secimi
- metrik tablolari
- absolute difference
- Fourier spectrum
- method ranking

eklenmistir.

Bu kisim bana, iyi bir algoritmanin tek basina yeterli olmadigini; gozlenebilirlik, debug edilebilirlik ve kullanici kontrollu arayuzun da proje basarisinin parcasi oldugunu ogretti.

### 4.12 `webcam_demo.py`

OpenCV tabanli basit bir real-time demo sunar. Bu dosya Streamlit disi bir test ortamidir. Yani GUI sorunlarini algoritma sorunlarindan ayirmaya yardimci olur.

---

## 5. Projede Kendi Yazdigim Mantigin Adim Adim Aciklamasi

Bu projede asil muhendislik degeri, farkli kutuphaneleri kullanmis olmak degil; onlarin arasina **dogru veri akisini kurmus olmamdir**.

### Adim 1: Girdi alma

Kullanici bir JPG/PNG secer. Sistem once dosyanin varligini ve uzantisini kontrol eder. Sonra dosya pikselleri okunur ve metadata ile birlikte bir `dict` icerisinde tasinir.

Neden `dict` kullandim?

- tek basina `numpy.ndarray` yeterli degildi
- `color_space`, `width`, `height`, `dtype`, `file_name` gibi bilgiler pipeline boyunca gerekiyordu

### Adim 2: Standardizasyon

Girdi resmi once preprocessing ile 512x512 standardina cektim. Bunu yapma sebebim:

- detector, FFT ve metrikler icin ortak bir olcek kullanmak
- ornekler arasi karsilastirmayi anlamli hale getirmek

### Adim 3: ROI secimi

Tum goruntu yerine sadece yuz bolgesini sectim. Bu adim hesaplama maliyetini dusurdu ve daha kararlı landmark cikisi sagladi.

### Adim 4: Landmark temsil

MediaPipe Face Mesh ile yuzu 468 nokta uzerinden temsil ettim. Boylece artik yuz sadece resim degil, ayni zamanda geometri oldu.

### Adim 5: Target geometri olusturma

Standard transformlarda belirli landmark index'lerini hareket ettirdim:

- gulumseme icin agiz koseleri
- kas kaldirma icin eyebrow noktalar
- lip widening icin agiz genisligi
- face slimming icin yanak/jawline

### Adim 6: Warp uygulama

Kaynak ve hedef landmark setleri arasinda Delaunay ucgenleri kurup affine warp uyguladim. Boylece lokal deformasyonlarla dogal yuzey surekliligini korumaya calistim.

### Adim 7: DSP analizi

Donusmus goruntunun Fourier spektrumunu hesaplayip yuksek/dusuk frekans oranlarina baktim. Bu, donusumlerin detay kaybi ya da asiri sertlestirme yapip yapmadigini inceleme imkani verdi.

### Adim 8: Nicel degerlendirme

`MSE`, `PSNR`, `SSIM`, `Absolute Difference` ile once-sonra farklarini olctum.

### Adim 9: Advanced expression transfer

Source ve reference landmarklarini ayni koordinat sistemine alip:

- ya landmark farklariyla
- ya TPS ile
- ya da ifade katsayilariyla

source yuz ifadesini degistirdim.

### Adim 10: Real-time optimizasyon

Live modda:

- frame downscale
- frame skip
- landmark EMA smoothing
- region tabanli transfer
- ayri kalite profilleri

kullanarak sistemi daha akici hale getirdim.

---

## 6. Projeden Uretilen Ornek Artefact'lar

### 6.1 Ham giris ornegi

![](images/sample-face.jpg)

### 6.2 Preprocessing sonucu

![](images/preprocessed-face.png)

### 6.3 Histogram

![](images/histogram.png)

### 6.4 Yuz tespiti

![](images/detected-face.png)

### 6.5 Face crop

![](images/face-crop.png)

### 6.6 Landmark overlay

![](images/landmarks-face.png)

### 6.7 Geometric warping cikti ornegi

![](../outputs/warping/test_face_1_smile_enhancement_comparison.png)

### 6.8 Aging cikti ornegi

![](../outputs/aging/test_face_1_aged.png)

### 6.9 Difference visualization ornegi

![](../outputs/evaluation/test_face_1_difference.png)

### 6.10 GUI'de expression transfer karsilastirma ornegi

![](../readme-comparing-transformed.png)

Bu goruntuler raporun sadece teorik kalmadigini, gercek output'lar urettigini gostermektedir.

---

## 7. Ornek Sayisal Sonuclar

`outputs/evaluation/test_face_1_metrics.csv` dosyasinda kayitli ornek degerler:

- `MSE = 162.2962`
- `PSNR = 26.0277`
- `SSIM = 0.9243`
- `Mean Absolute Difference = 3.2663`
- `Max Absolute Difference = 182`

`outputs/frequency/test_face_1_frequency_metrics.csv` dosyasindaki ornek Fourier olcumleri:

- `Total Energy = 2790510850736128.0`
- `Low Frequency Energy = 2784855184113664.0`
- `High Frequency Energy = 5654900113408.0`
- `High / Low Ratio = 0.00203059`
- `Low Frequency Radius = 51`

Bu sonuclar bana su yorumu yapma imkani verdi:

- donusum sonrasinda yapisal benzerlik yuksek kalmis,
- farklar lokal bolgelerde yogunlasmis,
- enerjinin buyuk kismi dusuk frekansta kalmaya devam etmis,
- yani donusum tum yuzu bozmak yerine daha cok lokal ifade bolgelerini etkilemis.

---

## 8. Real-Time Tarafinda Ogrendiklerim

Real-time kisim, offline pipeline'dan farkli olarak bana su gercegi ogretti:

Bir algoritma tek resimde iyi calisabilir ama kamerada:

- yavas olabilir,
- jitter uretebilir,
- UI tekrar render yuzunden donabilir,
- landmark'lar ziplayabilir.

Bu nedenle sistemde su yaklasimlari kullandim:

- `max_width` ile input downscale
- `frame_skip_interval` ile tum frame'leri tam islememe
- `previous_landmarks` ile EMA smoothing
- `quality profiles` ile kullaniciya hiz/kalite secimi verme

Bu, yazilim muhendisligi acisindan "aynı algoritma, farkli runtime kosulu" kavramini netlestirdi.

---

## 9. Test Altyapisi ve Muhendislik Kalitesi

Projede `tests/test_smoke.py` dosyasi icinde **36 adet test** bulunuyor.

Bu testler:

- image loading
- preprocessing
- face detection
- landmark export
- warping
- FFT
- frequency metrics
- aging/de-aging
- evaluation export
- reference expression transfer
- realtime frame pipeline

gibi farkli modulleri kapsiyor.

Bu kisim bana sunu ogretti:

- goruntu isleme projeleri de test edilebilir
- sadece son goruntuye bakmak yeterli degildir
- dosya ciktilari, boyutlar, metriklerin varligi ve veri yapilari otomatik denetlenebilir

---

## 10. Projenin Guclu Yanlari

Bu projenin muhendislik acisindan guclu buldugum yonleri:

- moduler mimari
- GUI ve core pipeline ayrimi
- output artefact kaydi
- sayisal metriklerle degerlendirme
- klasik warp ile modern transfer mantiginin ayni projede bulunmasi
- real-time ve offline akislarin birlikte dusunulmesi

---

## 11. Sinirlar ve Gelistirme Alanlari

Bu projenin mevcut sinirlari da vardir:

- Haar Cascade detector modern detector'ler kadar guclu degil
- landmark kalitesi occlusion ve pose degisimlerinde dusuyor
- klasik warping bazen identity bozulmasi uretebiliyor
- Streamlit real-time yapisi tam production seviyesinde degil
- expression transfer halen tam reenactment modeli kadar dogal degil

Gelecekte eklenebilecek daha guclu yonler:

- InsightFace veya daha guclu detector/alignment
- Kalman filter / optical flow tracking
- GPU hizlandirmasi
- coefficient tabanli 3D face model
- First Order Motion Model / SadTalker benzeri reenactment
- daha iyi blend ve illumination normalization

---

## 12. Sonuc

Bu proje sonunda yalnizca bir yuz filtresi yapmis olmadim; aslinda asagidaki yetkinlikleri ayni sistemde birlestirmis oldum:

- goruntu verisini dogrulama
- sinyal standardizasyonu
- ROI tabanli analiz
- landmark tabanli geometri modelleme
- affine ve TPS tabanli warping
- frekans uzayi analizi
- kalite metrikleriyle degerlendirme
- GUI ve real-time entegrasyonu
- test yazimi ve moduler tasarim

DSP dersi perspektifinden baktigimda bu proje bana su ana fikri ogretti:

**Bir goruntu sadece goruntu degildir; o, geometrisi, frekansi, enerjisi ve yapisal benzerligi olan sayisal bir sinyaldir.**

Bilgisayar muhendisligi perspektifinden baktigimda ise en buyuk kazanım su oldu:

**iyi bir proje, tek bir algoritmadan degil; dogru veri yapilarindan, moduler katmanlardan, testlerden, gozlenebilir output'lardan ve kullanilabilir arayuzden olusur.**

---

## 13. Raporda Referans Verilen Baslica Dosyalar

- `app.py`
- `streamlit_app.py`
- `gui/dashboard.py`
- `webcam_demo.py`
- `src/facial_image_warping/input_module.py`
- `src/facial_image_warping/preprocessing.py`
- `src/facial_image_warping/face_detection.py`
- `src/facial_image_warping/landmark_detection.py`
- `src/facial_image_warping/geometric_warping.py`
- `src/facial_image_warping/fourier_analysis.py`
- `src/facial_image_warping/aging_filter.py`
- `src/facial_image_warping/expression_transfer.py`
- `src/facial_image_warping/evaluation.py`
- `tests/test_smoke.py`

