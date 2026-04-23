# Mini Plant Set

# 🌿 Mini Plant Set

### 📌 Provenance

This dataset is composed of curated subsets pulled from public mirrors of the **PlantVillage** and **Cassava Leaf Disease** datasets.

### 📊 Dataset Composition

* **Clean Set (Training & Evaluation):** 1,500 labeled images.
* **Field Set (Robustness Testing):** 60 noisy 'field-shot' images featuring motion blur and mixed lighting, designed to test model resilience in real-world scenarios.

### ⚖️ Class Weights

The subset is perfectly balanced across all 5 target classes:

```json
{
  "healthy": 1.0,
  "maize_rust": 1.0,
  "maize_blight": 1.0,
  "cassava_mosaic": 1.0,
  "bean_spot": 1.0
}
```

    -**Provenance**: Subsets pulled from public mirrors of PlantVillage, Cassava Leaf Disease, and Beans datasets.
    - **Total Images**: 1,500 clean, 60 noisy field-shots.
    - **Class Weights**: {
  "healthy": 1.0,
  "maize_rust": 1.0,
  "maize_blight": 1.0,
  "cassava_mosaic": 1.0,
  "bean_spot": 1.0
}
