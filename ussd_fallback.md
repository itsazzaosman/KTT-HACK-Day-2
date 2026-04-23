# USSD & SMS Fallback Strategy: Crop Diagnostics

While the **Edge-AI API** provides high-speed diagnostics for agricultural extension officers with smartphones, this fallback mechanism ensures that farmers in low-connectivity areas (e.g., rural Rwanda) can still access diagnostic advice using **USSD** and **SMS** on basic 2G phones.

---

## 1. User Journey (USSD Menu)

Farmers dial a short code (e.g., `*801#`) to access the text-based diagnostic service.

| Step        | User Action    | Menu Display (Kinyarwanda / English)                                                                                                                                         |
| :---------- | :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | Dial `*801#` | 1. Diagnostika Igihingwa (Diagnose Crop)`<br>`2. Amakuru y'Igihe (Weather/Advice)                                                                                          |
| **2** | Select `1`   | Hitamo igihingwa (Select Crop):`<br>`1. Ibigori (Maize)`<br>`2. Imyumbati (Cassava)`<br>`3. Ibishyimbo (Beans)                                                         |
| **3** | Select `1`   | Sobanura ikibazo (Describe symptoms):`<br>`1. Amababi afite amabara y'umugese (Rust spots)`<br>`2. Amababi akanyaraye (Curled leaves)`<br>`3. Icyatsi kibisi (Healthy) |
| **4** | Select `1`   | **Result:** "Ibigori byawe bishobora kuba bifite 'Maize Rust'. Twaguhaye ubutumwa bugufi (SMS) burimo inama."                                                          |

---

## 2. Automated SMS Advice

Following the USSD session, the system automatically triggers an SMS containing the diagnosis rationale and recommended treatment.

**Example SMS (Maize Rust):**

> **KTT Diagnostics:** Twabonye ko ibigori byawe bifite amabara y’umugese (Maize Rust).
> **Inama:** Kura ibihingwa birwaye mu murima, ukoreshe ifumbire n'imiti yagenwe na RAB. Hamagara 1110 niba ukeneye ubufasha.

---

## 3. Implementation Workflow

1. **User Input:** Farmer describes symptoms via USSD text prompts.
2. **Logic Engine:** The backend maps symptom descriptions to the most likely labels identified by our MobileNetV3 model.
3. **Response:** The system fetches the corresponding `rationale` from our dictionary and translates it into Kinyarwanda/French for the farmer.

---

## 4. Key Benefits

* **Inclusion:** Works on any feature phone (e.g., Nokia 3310) with zero data balance.
* **Localization:** Provides instant advice in the farmer's primary language.
* **Offline Scalability:** Serves as a vital bridge in the "last mile" of agricultural technology.
