# Food Information API 
A Natural Language Processing (NLP) API that provides food information. This system can answer descriptive questions about food and provide food recommendations based on natural language queries.

## 📋 Table of Contents

- [Features](#features)
- [API Documentation](#api-documentation)
- [Dataset Information](#dataset-information)
- [Technical Implementation](#technical-implementation)
- [Setup and Installation](#setup-and-installation)
- [API Usage Examples](#api-usage-examples)

## ✨ Features

- Natural language processing for food-related queries
- Descriptive information about specific foods
- Food recommendations based on type and calorie preferences
- Cross-origin resource sharing (CORS) enabled

## 🚀 API Documentation

### Base URL

The API is served at:
- Local development: `http://localhost:5000`
- Default server: `http://192.168.43.163:5000`

### Endpoints

#### GET `/`

Returns basic API information.

**Response:**
```json
{
  "status": "success",
  "message": "Food Information API is running",
  "endpoints": {
    "/api/food": "POST - Process any food query (search, info, listing)"
  }
}
```

#### POST `/food` or `/api/food`

Processes natural language queries about food.

**Request Body:**
```json
{
  "query": "Apa itu nasi goreng?"
}
```

**Response (Descriptive Query):**
```json
{
  "status": "success",
  "query_type": "descriptive",
  "entities": {
    "food_type": null,
    "food_name": "nasi goreng",
    "keterangan_kalori": null
  },
  "answer": "Nasi goreng adalah makanan berupa nasi yang digoreng dan diaduk dalam minyak goreng, margarin, atau mentega. Jenis: makanan berat. Kalori: 267 kkal."
}
```

**Response (Recommendation Query):**
```json
{
  "status": "success",
  "query_type": "recommendation",
  "entities": {
    "food_type": "makanan ringan",
    "food_name": null,
    "keterangan_kalori": "kalori rendah"
  },
  "results": [
    {
      "name": "Roti Gandum",
      "calories": 80,
      "type": "makanan ringan",
      "calorie_status": "kalori rendah",
      "description": "Roti yang terbuat dari biji gandum utuh",
      "similarity": 0.85
    },
    ...
  ]
}
```

#### GET `/food` or `/api/food`

Returns a list of all foods or foods filtered by type.

**Query Parameters:**
- `type` (optional): Filter foods by type (e.g., "makanan berat", "makanan ringan", "cemilan", "minuman")

**Response:**
```json
{
  "status": "success",
  "count": 10,
  "type": "makanan berat",
  "foods": [
    {
      "name": "Nasi Goreng",
      "calories": 267,
      "type": "makanan berat",
      "calorie_status": "kalori sedang"
    },
    ...
  ]
}
```

## 📊 Dataset Information

### Structure

The dataset contains food information with the following attributes:

| Field | Description |
|-------|-------------|
| id | Unique identifier for each food item |
| nama_makanan | Food name |
| deskripsi | Food description |
| kategori | Food category (typically "makanan" or "minuman") |
| kalori | Calorie content (in kcal) |
| status_kesehatan | Health status (sehat, sedang, tidak sehat) |
| jenis | Food type (makanan berat, makanan ringan, cemilan, minuman) |
| keterangan_kalori | Calorie level category (Rendah, Sedang, Tinggi) |

### Food Categories

1. **Makanan Berat** - Main meals (e.g., Nasi Pecel, Soto Ayam, Bubur Ayam)
2. **Makanan Ringan** - Light meals (e.g., Tahu Gejrot, Tempe Goreng, Karedok)
3. **Cemilan** - Snacks (e.g., Getuk, Rujak Buah, Lupis)
4. **Minuman** - Beverages (e.g., Wedang Jahe, Es Kelapa Muda, Kopi Tubruk)

### Calorie Categories

Based on the dataset analysis:

1. **Kalori Rendah (Low Calorie)** - Foods with calorie content typically up to 200 kcal
   * Examples: Wedang Jahe (100 kcal), Sayur Bayam (150 kcal), Karedok (180 kcal)
   
2. **Kalori Sedang (Medium Calorie)** - Foods with calorie content typically between 201-400 kcal
   * Examples: Tempe Goreng (250 kcal), Gado-Gado (250 kcal), Soto Ayam (300 kcal)
   
3. **Kalori Tinggi (High Calorie)** - Foods with calorie content typically above 400 kcal
   * Examples: Rendang (400 kcal), Martabak Manis (450 kcal), Tumpeng (600 kcal)


## 💻 Technical Implementation

### information.py Components

#### Data Processing

- `setup_data()`: Loads the food dataset, initializes TF-IDF vectorizer, and builds the TF-IDF matrix
- `clean_text()`: Preprocesses text by removing punctuation, extra spaces, and converting to lowercase

#### Query Understanding

- `detect_query_type()`: Determines if a query is descriptive (asking about a specific food) or a search/recommendation query
- `extract_query_entities()`: Extracts key entities from queries, including food names, food types, and calorie preferences
- `enhanced_detect_query_type()`: Enhanced version with additional pattern matching

#### Information Retrieval

- `search_food_with_focal()`: Implements the FOCAL approach to retrieve relevant food information
- `answer_query()`: Generates descriptive answers about specific foods
- `get_most_similar_food()`: Finds the most similar food to a given query using cosine similarity

### Models and Algorithms

1. **TF-IDF Vectorization**:
   - Used for converting food descriptions into numerical vectors
   - Captures importance of terms relative to the corpus

2. **Cosine Similarity**:
   - Measures similarity between query vectors and food description vectors
   - Used for ranking and retrieving relevant food items

3. **Entity Extraction**:
   - Pattern matching with regular expressions
   - Keyword spotting for food types and calorie preferences

## 🔧 Setup and Installation

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Access the chatbot interface by opening `layout.html` in a web browser

## 📝 API Usage Examples

### Descriptive queries:

- "Apa itu nasi goreng?"
- "Berapa kalori dalam rendang?"
- "Jelaskan tentang soto ayam"

### Recommendation queries:

- "Rekomendasikan makanan ringan"
- "Makanan dengan kalori rendah"
- "Berikan contoh cemilan"
- "Saya ingin minuman dengan kalori tinggi"

---

Developed as part of the Natural Language Processing project.
