from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from information import (
    setup_data, 
    answer_query, 
    search_food_with_focal, 
    extract_query_entities, 
    detect_query_type
)
import os
import pandas as pd
import re

app = Flask(__name__)
# Configure CORS more thoroughly
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Initialize data on startup
print("Initializing Food Information System...")
df, vectorizer, tfidf_matrix = setup_data()
print(f"Successfully loaded {len(df)} food records")

def clean_answer_format(text):
    """Remove newlines and format the answer to be clean for API responses"""
    # Replace all newline sequences with spaces
    text = re.sub(r'\n+', ' ', text)
    # Clean up any multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # Fix bullet points
    text = text.replace('- Jenis:', ' Jenis:')
    text = text.replace('- Kalori:', ' Kalori:')
    return text.strip()

# Enhanced query type detection to handle more patterns
def enhanced_detect_query_type(query):
    """Enhanced detection of query type to handle more patterns"""
    query = query.lower()
    
    # Call the original detect_query_type first
    original_type = detect_query_type(query)
    
    # If already detected as descriptive, return that
    if original_type == 'descriptive':
        return original_type
    
    # Additional descriptive patterns with inverted structure - more flexible patterns
    inverted_descriptive_patterns = [
        r'([a-zA-Z\s]+)\s+itu\s+(apa|bagaimana|gimana)',  # "martabak manis itu apa?"
        r'(kalori|gizi|nutrisi|kandungan)\s+(?:dalam|untuk|pada|di)\s+([a-zA-Z\s]+)',  # "kalori dalam nasi goreng"
        r'([a-zA-Z\s]+)\s+(?:termasuk|masuk|merupakan)\s+(?:jenis|kategori|makanan)\s+(apa)'  # "nasi goreng termasuk jenis apa"
    ]
    
    # Check for inverted descriptive patterns
    for pattern in inverted_descriptive_patterns:
        match = re.search(pattern, query)
        if match:
            print(f"Matched inverted descriptive pattern: '{match.group(0)}'")
            return 'descriptive'
    
    # Check if the query exactly matches a food name
    for food_name in df['nama_makanan'].str.lower():
        if query == food_name or query == food_name + "?":
            print(f"Query matches exact food name: '{food_name}'")
            return 'descriptive'
    
    # Additional check for queries that contain food names with few extra words
    food_names = set(df['nama_makanan'].str.lower())
    
    # Check if any food name is a substantial part of the query
    for food in sorted(food_names, key=len, reverse=True):  # Check longer names first
        if food in query and len(food) > 3:  # Only consider meaningful food names
            words_in_food = len(food.split())
            words_in_query = len(query.split())
            
            # If the food name is a significant portion of the query
            if words_in_food / words_in_query > 0.5:
                print(f"Query contains significant food name: '{food}'")
                return 'descriptive'
    
    # Otherwise keep the original detection
    return original_type

# Additional function to better extract food type from query
def extract_better_food_type(query, entities):
    """Improve food type extraction from the query"""
    query = query.lower()
    
    # Map of food types we want to detect - matching the patterns in information.py
    food_types = {
        'makanan berat': ['makanan berat', 'berat'],
        'makanan ringan': ['makanan ringan', 'ringan'],
        'cemilan': ['cemilan', 'camilan', 'snack'],
        'minuman': ['minuman', 'minum', 'drink', 'jus', 'susu']
    }
    
    # First check for exact food types
    for food_type, keywords in food_types.items():
        for keyword in keywords:
            if keyword in query:
                return food_type
    
    # Check for invalid food_type values and clean them
    if entities['food_type']:
        # Check if it seems to be a command rather than a food type
        invalid_words = ['berikan', 'cari', 'rekomendasi', 'tampil', 'tunjuk', 'mau', 'ingin', 'saya']
        if any(word in entities['food_type'] for word in invalid_words):
            return None
    
    return entities['food_type']

# Add an explicit OPTIONS route handler for /food endpoint
@app.route('/food', methods=['OPTIONS'])
@cross_origin()
def handle_food_options():
    return '', 200

# Add a route handler for /food (notice no "api" prefix)
@app.route('/food', methods=['POST', 'GET'])
@cross_origin()
def food_endpoint():
    # Redirect to the same functionality as /api/food
    return food_api()

# Add a route handler for preflight OPTIONS requests
@app.route('/api/food', methods=['OPTIONS'])
@cross_origin()
def handle_options():
    return '', 200

@app.route('/')
@cross_origin()
def index():
    return jsonify({
        "status": "success",
        "message": "Food Information API is running",
        "endpoints": {
            "/api/food": "POST - Process any food query (search, info, listing)",
        }
    })

@app.route('/api/food', methods=['POST', 'GET'])
@cross_origin()
def food_api():
    # Handle different request methods
    if request.method == 'GET':
        # Return list of all foods or foods by type
        food_type = request.args.get('type', '')
        
        if food_type:
            # Handle common food type queries
            if food_type.lower() == 'berat':
                food_type = 'makanan berat'
            elif food_type.lower() == 'ringan':
                food_type = 'makanan ringan'
            
            # Filter foods by type
            filtered_foods = df[df['jenis'].str.contains(food_type, case=False)]
            
            if filtered_foods.empty:
                return jsonify({
                    "status": "success",
                    "count": 0,
                    "foods": []
                })
            
            # Convert to list of dictionaries
            foods_list = []
            for _, row in filtered_foods.iterrows():
                foods_list.append({
                    "name": row['nama_makanan'],
                    "calories": int(row['kalori']),
                    "type": row['jenis'],
                    "calorie_status": row['keterangan_kalori']
                })
            
            return jsonify({
                "status": "success",
                "count": len(foods_list),
                "type": food_type,
                "foods": foods_list
            })
        else:
            # Return all food names
            foods = df['nama_makanan'].tolist()
            return jsonify({
                "status": "success",
                "count": len(foods),
                "foods": foods
            })
    
    else:  # POST method - process query
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "No query provided"
            }), 400
        
        query_text = data['query']
        
        # Use enhanced query type detection
        query_type = enhanced_detect_query_type(query_text)
        print(f"Query: '{query_text}' detected as {query_type}")
        
        # Extract query entities for additional information
        entities = extract_query_entities(query_text)
        
        # Fix food_type extraction
        entities['food_type'] = extract_better_food_type(query_text, entities)
        
        if query_type == 'descriptive':
            # Get answer for descriptive query
            answer = answer_query(query_text, df, vectorizer, tfidf_matrix)
            # Clean the answer format
            answer = clean_answer_format(answer)
            
            # Check if it's a "not found" response
            if "Maaf, saya tidak menemukan informasi" in answer:
                return jsonify({
                    "status": "success",
                    "query_type": "descriptive",
                    "entities": entities,
                    "answer": "Maaf, Data yang anda cari tidak ditemukan."
                })
            
            return jsonify({
                "status": "success",
                "query_type": "descriptive",
                "entities": entities,
                "answer": answer
            })
        else:
            # Handle as a search query
            results = search_food_with_focal(query_text, df, vectorizer, tfidf_matrix)
            
            # Check if results contain the "not found" message
            if len(results) == 1 and (results.iloc[0]['nama_makanan'] == "Tidak ditemukan" or results.iloc[0]['nama_makanan'] == "Error"):
                # Return a not found message with empty results array
                return jsonify({
                    "status": "success",
                    "query_type": "recommendation",
                    "entities": entities,
                    "results": [],
                    "message": "Maaf, Data yang anda cari tidak ditemukan."
                })
            
            # If results are empty, return a standard message
            if len(results) == 0:
                return jsonify({
                    "status": "success",
                    "query_type": "recommendation",
                    "entities": entities,
                    "results": [],
                    "message": "Maaf, Data yang anda cari tidak ditemukan."
                })
            
            # If results are empty but we have food_type entity, try direct filtering
            if len(results) == 0 and entities['food_type'] is not None:
                print(f"No results found with standard search. Trying direct filter with food type: {entities['food_type']}")
                
                # Direct filtering based on food type
                food_type = entities['food_type']
                filtered_df = df[df['jenis'].str.contains(food_type, case=False)]
                    
                if not filtered_df.empty:
                    # Add a default similarity score for sorting
                    filtered_df = filtered_df.copy()
                    filtered_df['similarity'] = 1.0
                    
                    # If we have keterangan_kalori, apply that filter too
                    if entities['keterangan_kalori'] and 'kalori' in entities['keterangan_kalori']:
                        ket_kalori = entities['keterangan_kalori'].split()[0]  # Get 'rendah'/'sedang'/'tinggi'
                        kal_filtered = filtered_df[filtered_df['keterangan_kalori'].str.contains(ket_kalori, case=False)]
                        if not kal_filtered.empty:
                            filtered_df = kal_filtered
                    
                    results = filtered_df.sort_values(by=['similarity', 'nama_makanan'], ascending=[False, True]).head(5)
            
            # Convert results to serializable format
            food_results = []
            for _, row in results.iterrows():
                # Skip the "Tidak ditemukan" placeholder if it somehow got here
                if row['nama_makanan'] == "Tidak ditemukan" or row['nama_makanan'] == "Error":
                    continue
                    
                food_results.append({
                    "name": row['nama_makanan'],
                    "calories": int(row['kalori']),
                    "type": row['jenis'],
                    "calorie_status": row['keterangan_kalori'],
                    "description": row['deskripsi'],
                    "similarity": float(row['similarity'])
                })
                
            return jsonify({
                "status": "success",
                "query_type": "recommendation",
                "entities": entities,
                "results": food_results
            })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    ###