Here is the English translation of the annotation guideline document:

---

# Medical Entity Recognition Task Rules

## Annotation Guidelines

Below is a summary of all fields that need to be extracted:

| Entity Type | Description |
| --- | --- |
| Disease / Symptom | Disease terms and symptom terms. Symptom terms broadly include signs, abnormal test/exam findings, etc. Do not include unnecessary degree adverbs/modifiers. Examples: chronic cough, cervical disc herniation, allergic dermatitis, acute laryngitis, bladder stones, hypertension, angina, dizziness, headache, visual fatigue, gynecological diseases, hypothyroidism, pulmonary bullae. |
| Lab Test / Examination | Laboratory test items, examination items, or indicator terms. Examples: liver function, complete blood count, CT scan, fasting blood glucose. |
| Surgery / Procedure | Surgical or medical procedure terms. Examples: orthodontics, strabismus surgery, rhinoplasty, teeth cleaning, ICL lens implantation surgery. |
| Organ / Body Part | Organ or body part terms. Examples: eyes, liver, fingers. If the organ term is already included within another entity type, do not extract the organ separately. (For example, for “headache,” only extract the disease/symptom entity “headache,” not “head.”) |
| Province | Such as Zhejiang, Zhejiang Province. |
| City | Such as Beijing, Beijing Municipality. |
| District / County | Such as Yuhang, Huangpu District, Kunshan City, Tonglu. |
| Hospital | Hospital names, such as Shanghai East Hospital. Note: if the hospital name contains a province, city, or district/county, those should also be extracted in the corresponding Province / City / District-County fields. |
| Hospital Type | Types of hospitals, such as tertiary hospital, secondary hospital, general hospital, public hospital, cancer hospital, traditional Chinese medicine hospital, dermatology hospital, etc. These indicate a category/type of hospital. |
| Department | Such as gastroenterology, obstetrics and gynecology. Do not extract department names again if they are already included within another entity type. (For example, for “Hangzhou Orthopedic Hospital,” extract Hospital = “Hangzhou Orthopedic Hospital” and City = “Hangzhou,” but do not extract “orthopedics” as a department.) |
| Person Name | Such as Jay Chou, Jack Ma. Do not extract a person name again if it is already part of another entity type. (For example, for “Ningbo Li Huili Hospital,” extract Hospital = “Ningbo Li Huili Hospital” and City = “Ningbo,” but do not extract “Li Huili” as a person name.) |
| Doctor Gender | Expressions such as male doctor, female doctor. |
| Doctor Surname | Expressions such as Dr. Zhang, Doctor Wang, Director Zhang. If a full person name has already been extracted, do not also extract the surname. (For example, for “Dr. Zhong Nanshan,” extract Person Name = “Zhong Nanshan,” not surname “Zhong” / “Dr. Zhong.”) |
| Drug Ingredient Name | Chemical ingredient names, such as ibuprofen, penicillin, amoxicillin. |
| Generic Drug Name | Usually a combination of chemical ingredient + dosage form, such as ibuprofen sustained-release capsules, amoxicillin clavulanate potassium tablets. Also includes fairly clear and unambiguous abbreviated generic names, such as Ganmaoling, Banlangen, Lianhua Qingwen. |
| Brand/Trade Drug Name | Usually the commercial/trade name of a drug product, such as Fenbid, Lipitor, Amoxil. |
| Pharmaceutical Brand Name | Usually the manufacturer/company brand name, such as GSK, Bayer. |
| Chinese Medicinal Material | Such as angelica root, codonopsis, Smilax glabra, Ophiopogon japonicus. |
| Other Drug | Medicines that cannot be classified into the above drug categories, such as cold medicine, antibiotics, NSAIDs, antihypertensive drugs. |
| Health Supplement | Such as calcium-zinc gel candy, vitamin B complex tablets, Naobaijin, By-Health. (Health supplements may be difficult to subdivide into more specific categories such as generic name, ingredient name, trade name, etc., so they are grouped together.) |
| Foods for Special Medical Purposes (FSMP) | Such as nutritionally complete formula for special medical use. (These may also be difficult to subdivide further, so they are grouped together.) |
| Medical Device | Such as thermometer, sphygmomanometer, Omron. (Medical devices may be difficult to subdivide into more specific categories such as generic/device name vs. brand, so they are grouped together.) |
| Vaccine | Vaccine terms, such as influenza vaccine, 9-valent vaccine, Sinovac vaccine, etc. (Vaccines may also be difficult to subdivide further, so they are grouped together.) |