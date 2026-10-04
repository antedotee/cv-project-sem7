# Literature survey: Pediatric pneumonia classification from chest X-rays

## Abstract

Pediatric pneumonia classification from chest X-rays uses three labels: normal, bacterial pneumonia, and viral pneumonia. Bacterial pneumonia needs antibiotics. Viral pneumonia needs supportive care. A binary pneumonia label does not give that decision.

Kermany et al. released 5,856 frontal pediatric chest X-rays in 2018. Guangzhou Women and Children's Medical Center collected the images. The children are aged 1 to 5 years. This dataset is the common benchmark for the ten papers in this survey.

The ten papers show that binary accuracy on this dataset is often above 90%. Three-class accuracy is lower. Viral pneumonia is the weakest class. Papers that use the official test of 624 patients report 86% to 95% accuracy. Papers that use a custom 80/10/10 split report about 99%. One pipeline falls from 98.81% on this dataset to 86.95% on RSNA data.

**Index terms:** pediatric pneumonia, chest X-ray, Kermany dataset, convolutional neural network (CNN), transfer learning, class imbalance.

## Dataset

The Kermany dataset contains anterior-posterior chest X-rays from routine care. Two expert physicians graded the images. A third physician checked the evaluation set. The authors removed low-quality and unreadable images.

Official split from the *Cell* paper:

| Split | Normal | Bacterial | Viral | Total |
| --- | --- | --- | --- | --- |
| Train | 1,349 | 2,538 | 1,345 | 5,232 images |
| Test | 234 | 242 | 148 | 624 patients |

Pneumonia outnumbers normal about 3:1. Bacterial pneumonia outnumbers viral pneumonia. The test is patient-independent.

Kaggle later puts 16 train images in a validation folder. That folder is not in the *Cell* protocol. Do not treat papers with the official 624-patient test and papers with an 80/10/10 split as the same exam.

## Key papers

### 1. Kermany et al., 2018

**Identifying medical diagnoses and treatable diseases by image-based deep learning.**  
Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, et al. *Cell* 172(5):1122–1131.e2.  
<https://doi.org/10.1016/j.cell.2018.02.010>

- **Task:** Two binary tasks. Pneumonia versus normal. Bacterial versus viral. No three-class task.
- **Method:** Inception V3 with ImageNet weights. Convolutional layers frozen. Only the softmax layer trained. No class weights. No geometric augmentation.
- **Results:** Pneumonia versus normal: accuracy 92.8%, sensitivity 93.2%, specificity 90.1%, AUC 96.8%. Bacterial versus viral: accuracy 90.7%, sensitivity 88.6%, specificity 90.9%, AUC 94.0%.
- **Limit:** One hospital. Ages 1–5. No three-class accuracy. When they unfroze the convolutional stack, accuracy decreased.

The 90.1% figure is specificity, not precision. Later tables sometimes mislabel it.

### 2. Rajaraman et al., 2018

**Visualization and interpretation of convolutional neural network predictions in detecting pneumonia in pediatric chest radiographs.**  
Sivaramakrishnan Rajaraman, Sema Candemir, Incheol Kim, George Thoma, Sameer Antani. *Appl. Sci.* 8(10):1715.  
<https://doi.org/10.3390/app8101715>

- **Task:** Two binary tasks plus three-class. Official test of 624 patients.
- **Method:** Custom VGG16 with ImageNet weights. Last layer removed. Global average pooling and a dense head added. Anatomical-atlas crop of the lungs. Input 1024×1024. Mean normalization. No geometric augmentation. Noisy train images kept. CAM and LIME for localization.
- **Results:** Cropped pneumonia versus normal: accuracy 96.2%, AUC 0.993, recall 96.2%, F1 0.970. Bacterial versus viral: accuracy 93.6%, AUC 0.962, recall 98.4%, specificity 86.0%. Three-class: accuracy 91.8%, AUC 0.939, recall 90.0%, F1 0.910.
- **Limit:** The atlas crop under-segments costophrenic angles. Residual and Inception variants were over-parameterized on this small dataset.

Lung crop beats the full image.

### 3. Stephen et al., 2019

**An efficient deep learning approach to pneumonia classification in healthcare.**  
Okeke Stephen, Mangal Sain, Uchenna Joseph Maduh, Do-Un Jeong. *J. Healthc. Eng.* 2019:4180949.  
<https://doi.org/10.1155/2019/4180949>

- **Task:** Pneumonia versus normal only.
- **Method:** Four-block CNN without ImageNet weights. Best input 200×200×3. Keras augmentation: rotation 40°, shift/shear/zoom 0.2, horizontal flip. No class weights. No lung crop.
- **Results:** Train accuracy 95.31%. Validation accuracy 93.73%. AUC and sensitivity not reported.
- **Limit:** Custom split of 3,722 train and 2,134 validation. This is not the official test of 624 patients. Binary only.

A CNN without ImageNet weights can still classify this binary task.

### 4. Rahman et al., 2020

**Transfer learning with deep convolutional neural network (CNN) for pneumonia detection using chest X-ray.**  
Tawsifur Rahman, Muhammad E. H. Chowdhury, et al.  
<https://arxiv.org/abs/2004.06578>

- **Task:** Two binary tasks plus three-class.
- **Method:** AlexNet, ResNet18, DenseNet201, SqueezeNet. DenseNet201 is best on all three tasks. MATLAB. Five-fold. Rotation 45°. Scale and shift 10%. Each train class augmented to 4,500 images. Input 227×227 or 224×224.
- **Results:** DenseNet201 pneumonia versus normal: accuracy 98.0%, recall 99%, AUC 0.98, F1 0.981. Bacterial versus viral: accuracy 95.0%, recall 96%, AUC 0.952. Three-class: accuracy 93.3%, recall 93.2%, AUC 0.95.
- **Limit:** Own test of about 200 images per class. Not the official 624-patient test. Confusion matrices show bacterial and viral swap. Comparison table labels Kermany 90.1% as precision. The *Cell* paper reports specificity.

### 5. Salehi et al., 2021

**Automated detection of pneumonia cases using deep transfer learning with paediatric chest X-ray images.**  
Mohammad Salehi, Reza Mohammadi, Hamed Ghaffari, Nahid Sadighi, Reza Reiazi. *Br. J. Radiol.* 94:20201263.  
<https://doi.org/10.1259/bjr.20201263>

- **Task:** Pneumonia versus normal only.
- **Method:** VGG19, DenseNet121, Xception, ResNet50. DenseNet121 is best. SMOTE for imbalance. Rescale 1/255. Rotation 15°. Small shift, shear, zoom, and flips. Denoise, contrast, mean subtract.
- **Results:** DenseNet121: accuracy 86.8%, AUC 0.86. Xception: accuracy 86.0%, AUC 0.81. All four models: accuracy at least 83%. Sensitivity above 91%.
- **Limit:** Binary only. One hospital. Same-patient images may leak across splits. The authors state that published accuracy is likely an overestimate. Lateral views are absent.

### 6. Ayan, Karabulut, and Ünver, 2021

**Diagnosis of pediatric pneumonia with ensemble of deep convolutional neural networks in chest X-ray images.**  
Enes Ayan, Bergen Karabulut, Halil Murat Ünver. *Arab. J. Sci. Eng.* 47:2123.  
<https://doi.org/10.1007/s13369-021-06127-z>

- **Task:** Binary plus three-class. Test of 624 images.
- **Method:** VGG, ResNet, Inception, Xception, MobileNet, SqueezeNet, and PNet. Best ensemble uses three models, not seven. Normal images augmented from 1,349 to 2,534. Global average pooling, batch-norm, L2.
- **Results:** Binary ensemble: accuracy 95.83%, sensitivity 97.76%, specificity 92.73%, AUC 95.21%. Three-class ensemble: accuracy 90.71%. Bacterial recall 97.93%. Viral recall 77.03%. Viral F1 82.31%.
- **Limit:** Viral pneumonia remains the hard class. Larger ensembles decreased accuracy. They set hyperparameters by trial and error.

### 7. Kundu et al., 2021

**Pneumonia detection in chest X-ray images using an ensemble of deep learning models.**  
Rohit Kundu, Ritacheta Das, Zong Woo Geem, Gi-Tae Han, Ram Sarkar. *PLoS One* 16(9):e0256630.  
<https://doi.org/10.1371/journal.pone.0256630>

- **Task:** Pneumonia versus normal. Kermany dataset and RSNA as an external set.
- **Method:** Weighted-probability ensemble of GoogLeNet, ResNet-18, and DenseNet-121. Transfer learning. Resize 224×224×3. No SMOTE. No class weights.
- **Results:** Kermany: accuracy 98.81%, precision 98.82%, recall 98.80%, F1 98.79%, AUC 98.35%. RSNA: accuracy 86.95%, recall 87.02%.
- **Limit:** Binary only. Triple training cost. The ensemble fails on low-contrast images and early infiltrates. Accuracy falls about 12 points on the external hospital set.

### 8. XCCNet (Raghaw et al.), 2024

**An explainable contrastive-based dilated convolutional network with transformer for pediatric pneumonia detection.**  
C. S. Raghaw, P. S. Bhore, M. Z. U. Rehman, N. Kumar.  
<https://arxiv.org/abs/2410.16143>

- **Task:** Pneumonia versus normal.
- **Method:** Dilated CNN fused with a contrastive transformer. CLAHE. Lung segmentation with ResUNet++. Rib suppression. GAN synthesis of 2,624 extra normal images (1,583 to 4,207). Grad-CAM, Score-CAM, LIME.
- **Results:** No augmentation: accuracy 96.29%. Traditional augmentation: accuracy 97.18%. Full stack: accuracy 99.76%, precision/recall/F1 99.75%. Also tested on VinDr-PCXR, NIH-Pediatric, and Trivedi.
- **Limit:** The split is 80/10/10. This is not the official 624-patient test. The method needs labeled lungs and ribs. There is no severity grade.

The 99.76% result is not comparable to Kermany 92.8%. The tests differ.

### 9. Chauhan, Gupta, and Doja, 2025

**LightPneumoNet: Lightweight pneumonia classifier.**  
Neilansh Chauhan, Piyush Kumar Gupta, Faraz Doja.  
<https://arxiv.org/abs/2510.11232>

- **Task:** Pneumonia versus normal. Independent test.
- **Method:** Four-block CNN without ImageNet weights. 388,082 parameters. Size 1.48 MB. Filters 16 to 128. Input 224×224 grayscale in [0, 1]. Rotation 12°. Zoom, shift, and shear 0.15. No horizontal flip. Class weights: normal 2.0, pneumonia 1.2.
- **Results:** Accuracy 94.2%. Precision 0.92. Recall 0.99. F1 0.96.
- **Limit:** Binary only. They trade precision of 0.92 for recall of 0.99.

### 10. Choudhury, 2026

**Pediatric pneumonia detection from chest X-rays: A comparative study of transfer learning and custom CNNs.**  
A. R. Choudhury.  
<https://arxiv.org/abs/2601.00837>

- **Task:** Pneumonia versus normal.
- **Method:** Four-block CNN without ImageNet weights, versus ResNet50, DenseNet121, and EfficientNet-B0. Each backbone is frozen or fine-tuned on the last two blocks. Input is 224×224. Grayscale is copied to three channels. They use ImageNet mean and standard deviation. Train augmentation is flip, ±10° rotation, affine, and color jitter. No class weights. They report a train ratio of 2.87:1. Seed 42. Grad-CAM on true and false cases.
- **Results:** ResNet50 fine-tune: accuracy 99.43%, F1 99.61%, AUC 99.93%, sensitivity 99.48%, specificity 99.26%. The model makes three errors in 523 test images. Fine-tune beats frozen weights by 5.48% average accuracy. An ensemble does not increase accuracy after ResNet50 fine-tune.
- **Limit:** Stratified 80/10/10 from 5,216 train images (4,172 / 521 / 523). This is not the official 624-patient test. Binary only. No external hospital test. One hospital.

## Comparison

Do not rank these papers on one accuracy board. Task, imbalance method, and split differ.

| # | Paper | Task | Imbalance method | Reported accuracy | Split |
| --- | --- | --- | --- | --- | --- |
| 1 | Kermany | 2 binary | none | 92.8 / 90.7 | 624 patients |
| 2 | Rajaraman | 2 binary + 3-class | noisy train kept | 96.2 / 93.6 / 91.8 | 624 patients |
| 3 | Stephen | 1 binary | heavy augmentation | 93.7 validation | custom |
| 4 | Rahman | 2 binary + 3-class | 4,500 per class | 98 / 95 / 93.3 | ~200 per class |
| 5 | Salehi | 1 binary | SMOTE | 86.8 | held-out, leakage warned |
| 6 | Ayan | binary + 3-class | augment normal | 95.8 / 90.7 (viral recall 77) | ~624 |
| 7 | Kundu | 1 binary | transfer learning | 98.8, then 87 on RSNA | 2 datasets |
| 8 | XCCNet | 1 binary | GAN normals | 99.76 | 80/10/10 |
| 9 | LightPneumoNet | 1 binary | class weights | 94.2 (recall 0.99) | independent test |
| 10 | Choudhury | 1 binary | none | 99.43 | 80/10/10 of train |

## Themes and consensus

Transfer-learning CNNs are the default method. Inception V3, VGG16, ResNet, DenseNet, EfficientNet, and Xception appear across the corpus. DenseNet121 or DenseNet201 is best in Rahman and Salehi. ResNet50 fine-tune is best in Choudhury 2026.

A CNN without ImageNet weights still works for the binary task. Stephen reports 93.73% validation accuracy. LightPneumoNet reports 94.2% accuracy and 0.99 recall.

Binary accuracy on the Kermany dataset is usually high. Three-class accuracy is 4 to 8 points lower. Viral recall is the weak number. Kermany already shows bacterial-versus-viral accuracy of 90.7% against pneumonia-versus-normal accuracy of 92.8%. Ayan reports viral recall of 77.03% against bacterial recall of 97.93%.

Most papers handle class imbalance with augmentation:

- Geometric augmentation (Stephen)
- Minority oversampling of normal images (Ayan)
- Balance each class to 4,500 (Rahman)
- SMOTE (Salehi)
- Class weights (LightPneumoNet)
- GAN synthetic normals (XCCNet)

These ten papers do not use focal loss. Horizontal flip is not always valid. LightPneumoNet does not use horizontal flip because left and right anatomy differ.

Preprocessing ranges from resize and 1/255 to lung crop, CLAHE, and rib suppression. Rajaraman finds that lung crop beats the full image.

## Open questions and debates

**Official test versus custom split.** Papers that keep the 624-patient test report 86% to 95%. Papers that use an 80/10/10 split of train images report about 99%. Those results are not comparable.

**Binary versus three-class.** Binary pneumonia versus normal is easier to publish. Antibiotic choice depends on bacterial versus viral. Three-class and etiology numbers are lower and less often reported.

**Freeze versus fine-tune.** Kermany reports that Inception V3 accuracy decreased when they unfroze the convolutional layers. Choudhury reports a 5.48% average accuracy gain when they fine-tune the last blocks. These two protocols were not run on the same 624-patient test.

**Ensemble versus one network.** Ayan and Kundu use three networks. Choudhury finds that an ensemble adds no gain after ResNet50 fine-tune.

**Lung crop versus full image.** Rajaraman and XCCNet crop to lung. Most other papers do not crop.

**Cross-hospital generalization.** Kundu falls from 98.81% on Kermany to 86.95% on RSNA. Salehi and Choudhury list single-center data as a limit and remain in one hospital.

## Emerging trends

- 2018–2019: frozen Inception V3, VGG16 with lung crop, CNN without ImageNet weights.
- 2020–2021: ImageNet model zoo, three-task reporting, radiology-journal numbers, viral-class failure, RSNA external test.
- 2024–2026: CNN–transformer hybrid, GAN minority images, class weights, small networks, Grad-CAM and LIME, freeze-versus-fine-tune on a documented split.

Recent 99% figures use a reshuffled split, not the official 624-patient test. Recent papers also return to binary labels.

## Sources

1. Kermany et al., 2018. *Cell*. <https://doi.org/10.1016/j.cell.2018.02.010>. Data: <https://doi.org/10.17632/rscbjbr9sj.3>
2. Rajaraman et al., 2018. *Appl. Sci.* <https://doi.org/10.3390/app8101715>
3. Stephen et al., 2019. *J. Healthc. Eng.* <https://doi.org/10.1155/2019/4180949>
4. Rahman et al., 2020. arXiv:2004.06578. <https://arxiv.org/abs/2004.06578>
5. Salehi et al., 2021. *Br. J. Radiol.* <https://doi.org/10.1259/bjr.20201263>
6. Ayan et al., 2021. *Arab. J. Sci. Eng.* <https://doi.org/10.1007/s13369-021-06127-z>
7. Kundu et al., 2021. *PLoS One*. <https://doi.org/10.1371/journal.pone.0256630>
8. Raghaw et al., 2024. XCCNet. arXiv:2410.16143. <https://arxiv.org/abs/2410.16143>
9. Chauhan et al., 2025. LightPneumoNet. arXiv:2510.11232. <https://arxiv.org/abs/2510.11232>
10. Choudhury, 2026. arXiv:2601.00837. <https://arxiv.org/abs/2601.00837>
