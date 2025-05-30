# -*- coding: utf-8 -*-
"""tfidf_pusing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XgAh3cHPNuVv7qnwHt8JZ3Ea2ed_bwBn
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process  # Add fuzzy string matching
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import traceback  # Added for better error diagnostics

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocessing teks yang lebih baik dengan normalisasi, stemming, dan penanganan stopwords
    yang spesifik untuk domain makanan Indonesia
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Normalisasi karakter (misalnya é → e)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Tokenisasi
    tokens = word_tokenize(text)

    # Stopwords spesifik domain - simpan kata-kata penting terkait makanan
    food_domain_words = {'kalori', 'sehat', 'enak', 'gizi', 'lemak', 'protein', 'karbohidrat'}
    indo_stopwords = set(stopwords.words('indonesian')) - food_domain_words

    # Hapus stopwords
    tokens = [t for t in tokens if t not in indo_stopwords]

    return ' '.join(tokens)

# Local file loading instead of Google Colab upload
def load_data(file_path):
    """Load data from local CSV file"""
    print(f"Loading data from {file_path}...")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records")
        return df
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

# Specify a default file path (can be changed by user)
default_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "makanan_indonesia.csv")

# Function to handle the data loading and preprocessing
def setup_data(file_path=None):
    """Set up the data and TF-IDF matrices"""
    if file_path is None:
        file_path = default_data_path
        if not os.path.exists(file_path):
            print(f"Warning: Default file {file_path} not found.")
            alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "makanan.csv")
            if os.path.exists(alt_path):
                print(f"Using alternative file: {alt_path}")
                file_path = alt_path
            else:
                raise FileNotFoundError(f"Neither default file nor alternative file could be found.")
    
    # Load the data
    df = load_data(file_path)

    # Create combined text field
    df['combined_text'] = df['nama_makanan'].fillna('') + " "  +  df['jenis'].fillna('') + " "  + df['keterangan_kalori'].fillna('')

    # Terapkan preprocessing ke data
    df['combined_text_processed'] = df['combined_text'].apply(preprocess_text)

    # Buat TF-IDF yang lebih baik dengan parameter yang dioptimalkan
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),    # Gunakan unigram dan bigram untuk menangkap frasa
        sublinear_tf=True,     # Terapkan sublinear tf scaling (1 + log(tf))
        use_idf=True,          # Gunakan inverse document frequency
        norm='l2',             # Normalisasi vektor dengan L2 norm
        stop_words=stopwords.words('indonesian')  # Hapus stopwords Indonesia
    )

    # Fit dan transform data
    tfidf_matrix = vectorizer.fit_transform(df['combined_text_processed'])
    
    return df, vectorizer, tfidf_matrix

def extract_calorie_filter(query):
    numbers = list(map(int, re.findall(r'\d+', query)))
    query = query.lower()

    if "lebih dari" in query or "di atas" in query or "lebih besar dari" in query or "diatas" in query:
        if numbers:
            return numbers[0] + 1, None
    elif "kurang dari" in query or "di bawah" in query or "lebih kecil dari" in query or "dibawah" in query:
        if numbers:
            return None, numbers[0] - 1
    elif "antara" in query and len(numbers) >= 2:
        return min(numbers), max(numbers)
    elif len(numbers) == 1:
        return numbers[0], numbers[0]
    elif len(numbers) >= 2:
        return min(numbers), max(numbers)

    return None, None

def detect_intent(query):
    query = query.lower()
    if re.search(r"(apa itu|jelaskan|deskripsi|tentang|ceritakan|informasi|apa saja|bagaimana|mengapa|kenapa|apa yang|gimana|tolong jelaskan)", query):
        return "definisi"
    elif re.search(r"(berapa.*kalori|kalorinya|jumlah kalori|kandungan kalori|nilai kalori|total kalori)", query):
        return "kalori"
    elif re.search(r"(apakah.*sehat|sehatkah|status kesehatan|bagaimana kesehatannya|sehat atau tidak|baik untuk kesehatan)", query):
        return "status_kesehatan"
    elif re.search(r"(apa jenisnya|termasuk jenis apa|kategori|jenis makanan|kelompok)", query):
        return "jenis"
    elif re.search(r"(tinggi|rendah|keterangan kalori|level kalori)", query):
        return "keterangan_kalori"
    else:
        return "deskripsi"

def apply_filters(query, df):
    # Make a copy of the original dataframe
    filtered_df = df.copy()
    filters = {}

    # Cek apakah ada kata-kata yang sesuai dengan filter keterangan_kalori (rendah/sedang/tinggi)
    if re.search(r'\brendah kalori\b', query, re.IGNORECASE) or \
       (re.search(r'\brendah\b', query, re.IGNORECASE) and re.search(r'\bkalori\b', query, re.IGNORECASE)):
        filters["keterangan_kalori"] = "rendah"
    elif re.search(r'\bsedang kalori\b', query, re.IGNORECASE) or \
         (re.search(r'\bsedang\b', query, re.IGNORECASE) and re.search(r'\bkalori\b', query, re.IGNORECASE)):
        filters["keterangan_kalori"] = "sedang"
    elif re.search(r'\btinggi kalori\b', query, re.IGNORECASE) or \
         (re.search(r'\btinggi\b', query, re.IGNORECASE) and re.search(r'\bkalori\b', query, re.IGNORECASE)):
        filters["keterangan_kalori"] = "tinggi"

    # Cek apakah ada kata-kata yang sesuai dengan jenis makanan
    if re.search(r'\bmakanan berat\b', query, re.IGNORECASE) or re.search(r'\bberat\b', query, re.IGNORECASE):
        filters["jenis"] = "makanan berat"
    elif re.search(r'\bmakanan ringan\b', query, re.IGNORECASE) or re.search(r'\bringan\b', query, re.IGNORECASE):
        filters["jenis"] = "makanan ringan"
    elif re.search(r'\bcemilan\b', query, re.IGNORECASE) or re.search(r'\bcamilan\b', query, re.IGNORECASE) or re.search(r'\bsnack\b', query, re.IGNORECASE):
        filters["jenis"] = "cemilan"
    elif re.search(r'\bminuman\b', query, re.IGNORECASE) or re.search(r'\bdrink\b', query, re.IGNORECASE) or re.search(r'\bjus\b', query, re.IGNORECASE) or re.search(r'\bsusu\b', query, re.IGNORECASE):
        filters["jenis"] = "minuman"

    # Print filters for debugging
    print(f"Applied filters: {filters}")

    # Check if any filters were applied
    if not filters:
        print("No specific filters detected, using all data")
        return filtered_df  # Always returns a DataFrame, never None

    # First check each filter separately and collect matches
    successful_filters = {}
    for key, value in filters.items():
        mask = filtered_df[key].str.lower() == value.lower()
        matches = filtered_df[mask]

        print(f"- Filter {key}={value} matches {len(matches)} items")

        if len(matches) > 0:
            successful_filters[key] = value

    # Apply filters that have matches
    if successful_filters:
        print(f"Applying successful filters: {successful_filters}")
        for key, value in successful_filters.items():
            filtered_df = filtered_df[filtered_df[key].str.lower() == value.lower()]

        if len(filtered_df) == 0:
            print("Warning: No data matches the combination of successful filters")
            print("Trying more flexible approach...")

            # Try each filter individually and return a combined result
            combined_results = pd.DataFrame()
            for key, value in successful_filters.items():
                matches = df[df[key].str.lower() == value.lower()]
                if len(matches) > 0:
                    combined_results = pd.concat([combined_results, matches])

            if len(combined_results) > 0:
                print(f"Found {len(combined_results)} results using individual filters")
                return combined_results.drop_duplicates()

            print("Returning all data and relying on similarity scoring")
            return df
    else:
        print("Warning: No successful filters found")
        # Do a more flexible search - check for partial matches
        partial_matches = pd.DataFrame()
        for key, value in filters.items():
            # Check if column contains value (substring match)
            mask = filtered_df[key].str.lower().str.contains(value.lower())
            matches = filtered_df[mask]
            if len(matches) > 0:
                print(f"Found {len(matches)} partial matches for {key}={value}")
                partial_matches = pd.concat([partial_matches, matches])

        if len(partial_matches) > 0:
            print(f"Using {len(partial_matches.drop_duplicates())} partial matches")
            return partial_matches.drop_duplicates()

        print("Returning all data and relying on similarity scoring")
        return df

# Ekstraksi nama makanan dengan pendekatan yang lebih baik
def extract_food_name(query, food_names, vectorizer, tfidf_matrix):
    # Mencoba exact match terlebih dahulu
    for food in sorted(food_names, key=len, reverse=True):  # Prioritaskan nama yang lebih panjang
        if food.lower() in query.lower():
            return food

    # Jika tidak ada exact match, gunakan similarity tertinggi
    query_vec = vectorizer.transform([query])
    food_vecs = vectorizer.transform(food_names)
    similarities = cosine_similarity(query_vec, food_vecs).flatten()
    best_match_idx = similarities.argmax()

    # Hanya kembalikan jika similarity cukup tinggi
    if similarities[best_match_idx] > 0.3:
        return food_names[best_match_idx]

    return query



# Fungsi utama menjawab pertanyaan
def answer_query(query, df, vectorizer, tfidf_matrix):
    try:
        intent = detect_intent(query)
        food_names = df['nama_makanan'].tolist()
        food_name = extract_food_name(query, food_names, vectorizer, tfidf_matrix)

        # Cari makanan dengan similarity tertinggi
        tfidf_query = vectorizer.transform([food_name])
        cosine_sim = cosine_similarity(tfidf_query, tfidf_matrix).flatten()
        idx = cosine_sim.argmax()
        matched_row = df.iloc[idx]

        # Jika similarity terlalu rendah, kembalikan pesan tidak ditemukan
        if cosine_sim[idx] < 0.2:
            return f"Maaf, saya tidak menemukan informasi tentang '{food_name}'."

        # Handle missing fields with safe defaults
        deskripsi = "Tidak ada deskripsi tersedia"
        if 'deskripsi' in matched_row and not pd.isna(matched_row['deskripsi']):
            deskripsi = matched_row['deskripsi']

        kalori = 0
        if 'kalori' in matched_row and not pd.isna(matched_row['kalori']):
            kalori = matched_row['kalori']

        status_kesehatan = "tidak terkategorisasi"
        if 'status_kesehatan' in matched_row and not pd.isna(matched_row['status_kesehatan']):
            status_kesehatan = matched_row['status_kesehatan']

        jenis = "tidak terkategorisasi"
        if 'jenis' in matched_row and not pd.isna(matched_row['jenis']):
            jenis = matched_row['jenis']

        keterangan_kalori = "tidak terkategorisasi"
        if 'keterangan_kalori' in matched_row and not pd.isna(matched_row['keterangan_kalori']):
            keterangan_kalori = matched_row['keterangan_kalori']

        # Buat respons berdasarkan intent
        if intent == "definisi":
            return f"{matched_row['nama_makanan']}: {deskripsi}"
        elif intent == "kalori":
            return f"{matched_row['nama_makanan']} memiliki {kalori} kalori. {keterangan_kalori} kalori."
        elif intent == "status_kesehatan":
            return f"{matched_row['nama_makanan']} tergolong {status_kesehatan}."
        elif intent == "jenis":
            return f"{matched_row['nama_makanan']} termasuk jenis {jenis}."
        elif intent == "keterangan_kalori":
            return f"{matched_row['nama_makanan']} memiliki {keterangan_kalori} kalori ({kalori} kalori)."
        else:
            # Untuk pertanyaan deskriptif umum, berikan informasi lengkap
            response = f"{matched_row['nama_makanan']}: {deskripsi}\n\n"
            response += f"- Jenis: {jenis}\n"
            response += f"- Kalori: {kalori} ({keterangan_kalori} kalori)\n"
            return response
    except Exception as e:
        print(f"Error in answer_query: {str(e)}")
        return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."

def search_food(query, df, vectorizer, tfidf_matrix):
    """Pencarian makanan menggunakan cosine similarity standar"""
    min_cal, max_cal = extract_calorie_filter(query)

    try:
        # Apply filters to the data
        filtered_df = apply_filters(query, df)

        # Check if filtered_df is empty
        if filtered_df.empty:
            print("Tidak ada makanan yang cocok dengan kriteria filter Anda.")
            print("Mencoba pencarian dengan similarity saja tanpa filter...")
            # Fall back to using all data
            filtered_df = df.copy()

        # Transform query to vector space
        query_vec = vectorizer.transform([query])

        # Get similarities using cosine similarity
        similarities = cosine_similarity(query_vec,
                                       vectorizer.transform(filtered_df['combined_text_processed'])).flatten()

        # Add similarity scores to the dataframe
        filtered_df = filtered_df.copy()
        filtered_df['similarity'] = similarities

        # Apply calorie filters if specified
        if min_cal is not None and max_cal is None:
            calorie_df = filtered_df[filtered_df['kalori'] >= min_cal]
            if not calorie_df.empty:
                filtered_df = calorie_df
        elif min_cal is None and max_cal is not None:
            calorie_df = filtered_df[filtered_df['kalori'] <= max_cal]
            if not calorie_df.empty:
                filtered_df = calorie_df
        elif min_cal is not None and max_cal is not None:
            calorie_df = filtered_df[(filtered_df['kalori'] >= min_cal) & (filtered_df['kalori'] <= max_cal)]
            if not calorie_df.empty:
                filtered_df = calorie_df

        # Sort by similarity and return top results
        results = filtered_df.sort_values(by='similarity', ascending=False)

        # Ensure all fields exist with default values if missing
        results = results.copy()
        results['deskripsi'] = results['deskripsi'].fillna("Tidak ada deskripsi tersedia")
        return results[['nama_makanan', 'kalori', 'jenis', 'keterangan_kalori', 'deskripsi', 'similarity']].head(3)

    except Exception as e:
        print(f"Error dalam pencarian: {str(e)}")
        # Return empty DataFrame with the correct columns
        return pd.DataFrame(columns=['nama_makanan', 'kalori', 'jenis', 'keterangan_kalori', 'deskripsi', 'similarity'])

# Implementasi focal similarity untuk menangani data tidak seimbang
def focal_similarity(query_vec, doc_vecs, gamma=2.0):
    """
    Implementasi focal similarity untuk meningkatkan bobot pada dokumen yang relevan
    tetapi memiliki similarity lebih rendah (mengatasi data tidak seimbang)

    gamma: Parameter fokus, semakin tinggi nilainya semakin fokus pada contoh sulit
    """
    # Hitung cosine similarity biasa
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()

    # Terapkan focal weighting
    # (1-sim)^gamma * sim - memberikan bobot lebih pada dokumen dengan similarity menengah
    focal_weights = np.power(1 - similarities, gamma) * similarities

    return focal_weights

def search_food_with_focal(query, df, vectorizer, tfidf_matrix, gamma=2.0):
    """Pencarian makanan menggunakan focal similarity"""
    min_cal, max_cal = extract_calorie_filter(query)
    
    # Extract entities to better apply filters
    entities = extract_query_entities(query)

    try:
        # First check if we have a combination that doesn't exist in our dataset
        if entities['food_type'] and entities['keterangan_kalori']:
            # Check if the combination exists in our dataset
            combo_check = df[(df['jenis'].str.lower() == entities['food_type'].lower()) & 
                             (df['keterangan_kalori'].str.lower() == entities['keterangan_kalori'].lower())]
            
            if combo_check.empty:
                # Return standardized empty DataFrame with consistent message
                empty_result = pd.DataFrame({
                    'nama_makanan': ["Tidak ditemukan"],
                    'kalori': [0],
                    'jenis': [""],
                    'keterangan_kalori': [""],
                    'deskripsi': ["Maaf, Data yang anda cari tidak ditemukan."],
                    'similarity': [0.0]
                })
                return empty_result

        # Try to apply filters
        filtered_df = apply_filters(query, df)

        # Check if filtered_df is None or empty
        if filtered_df is None or (hasattr(filtered_df, 'empty') and filtered_df.empty):
            print("Tidak ada makanan yang cocok dengan kriteria filter Anda.")
            print("Mencoba pencarian dengan similarity saja tanpa filter...")
            # Fall back to using all data
            filtered_df = df.copy()
        
        # Apply explicit calorie status filter if specified in query
        if entities['keterangan_kalori'] is not None:
            print(f"Explicitly filtering by calorie status: {entities['keterangan_kalori']}")
            calorie_status_mask = filtered_df['keterangan_kalori'].str.lower() == entities['keterangan_kalori'].lower()
            calorie_filtered = filtered_df[calorie_status_mask]
            
            if not calorie_filtered.empty:
                filtered_df = calorie_filtered
                print(f"Applied calorie status filter, {len(filtered_df)} results remaining")
            else:
                print("No exact matches for calorie status, trying partial match")
                # Try partial match
                calorie_status_mask = filtered_df['keterangan_kalori'].str.lower().str.contains(entities['keterangan_kalori'].lower())
                calorie_filtered = filtered_df[calorie_status_mask]
                if not calorie_filtered.empty:
                    filtered_df = calorie_filtered
                    print(f"Applied partial calorie status filter, {len(filtered_df)} results remaining")

        # Use all data rows if filtered results are too few
        if len(filtered_df) < 5:
            print(f"Hanya {len(filtered_df)} hasil ditemukan dengan filter, menambahkan hasil serupa...")

        # Check if after filtering we have no results that match the specific type
        if entities['food_type'] and len(filtered_df) > 0:
            type_matches = filtered_df[filtered_df['jenis'].str.lower() == entities['food_type'].lower()]
            if type_matches.empty:
                # Return standardized empty DataFrame with consistent message
                empty_result = pd.DataFrame({
                    'nama_makanan': ["Tidak ditemukan"],
                    'kalori': [0],
                    'jenis': [""],
                    'keterangan_kalori': [""],
                    'deskripsi': ["Maaf, Data yang anda cari tidak ditemukan."],
                    'similarity': [0.0]
                })
                return empty_result

        query_vec = vectorizer.transform([query])

        # Gunakan focal similarity
        similarities = focal_similarity(query_vec,
                                      vectorizer.transform(filtered_df['combined_text_processed']),
                                      gamma=gamma)

        filtered_df = filtered_df.copy()
        filtered_df['similarity'] = similarities

        # Terapkan filter kalori - numerical range
        if min_cal is not None and max_cal is None:
            calorie_df = filtered_df[filtered_df['kalori'] >= min_cal]
            if not calorie_df.empty:
                filtered_df = calorie_df
        elif min_cal is None and max_cal is not None:
            calorie_df = filtered_df[filtered_df['kalori'] <= max_cal]
            if not calorie_df.empty:
                filtered_df = calorie_df
        elif min_cal is not None and max_cal is not None:
            calorie_df = filtered_df[(filtered_df['kalori'] >= min_cal) & (filtered_df['kalori'] <= max_cal)]
            if not calorie_df.empty:
                filtered_df = calorie_df

        # Double-check for keyword-based calorie filter one more time, 
        # in case it wasn't caught by the entity extraction
        if 'rendah kalori' in query.lower() or ('rendah' in query.lower() and 'kalori' in query.lower()):
            rendah_df = filtered_df[filtered_df['keterangan_kalori'].str.lower() == 'rendah']
            if not rendah_df.empty:
                print(f"Applied 'rendah' calorie filter, {len(rendah_df)} results found")
                filtered_df = rendah_df
        elif 'sedang kalori' in query.lower() or ('sedang' in query.lower() and 'kalori' in query.lower()):
            sedang_df = filtered_df[filtered_df['keterangan_kalori'].str.lower() == 'sedang']
            if not sedang_df.empty:
                print(f"Applied 'sedang' calorie filter, {len(sedang_df)} results found")
                filtered_df = sedang_df
        elif 'tinggi kalori' in query.lower() or ('tinggi' in query.lower() and 'kalori' in query.lower()):
            tinggi_df = filtered_df[filtered_df['keterangan_kalori'].str.lower() == 'tinggi']
            if not tinggi_df.empty:
                print(f"Applied 'tinggi' calorie filter, {len(tinggi_df)} results found")
                filtered_df = tinggi_df

        # If we have valid results
        if not filtered_df.empty:
            # Final check to see if results match the requested food type
            if entities['food_type']:
                matching_type = filtered_df[filtered_df['jenis'].str.lower() == entities['food_type'].lower()]
                if not matching_type.empty:
                    filtered_df = matching_type
                    print(f"Final filter applied for food type: {entities['food_type']}")
                else:
                    # Return standardized empty DataFrame with consistent message
                    empty_result = pd.DataFrame({
                        'nama_makanan': ["Tidak ditemukan"],
                        'kalori': [0],
                        'jenis': [""],
                        'keterangan_kalori': [""],
                        'deskripsi': ["Maaf, Data yang anda cari tidak ditemukan."],
                        'similarity': [0.0]
                    })
                    return empty_result
            
            # Sort results
            results = filtered_df.sort_values(by='similarity', ascending=False)
            # Ensure all fields exist with default values if missing
            results = results.copy()
            results['deskripsi'] = results['deskripsi'].fillna("Tidak ada deskripsi tersedia")
            
            # Return top 5 results
            return results[['nama_makanan', 'kalori', 'jenis', 'keterangan_kalori', 'deskripsi', 'similarity']].head(5)
        else:
            print("Tidak ada hasil yang cocok setelah menerapkan semua filter.")
            
            # Return standardized empty DataFrame with consistent message
            empty_result = pd.DataFrame({
                'nama_makanan': ["Tidak ditemukan"],
                'kalori': [0],
                'jenis': [""],
                'keterangan_kalori': [""],
                'deskripsi': ["Maaf, Data yang anda cari tidak ditemukan."],
                'similarity': [0.0]
            })
            return empty_result

    except Exception as e:
        print(f"Error dalam pencarian: {str(e)}")
        print(f"Query: '{query}'")
        
        # Return standardized empty DataFrame with consistent message
        empty_result = pd.DataFrame({
            'nama_makanan': ["Error"],
            'kalori': [0],
            'jenis': [""],
            'keterangan_kalori': [""],
            'deskripsi': ["Maaf, Data yang anda cari tidak ditemukan."],
            'similarity': [0.0]
        })
        return empty_result

# Function to detect query type
def detect_query_type(query):
    """
    Determines if the query is a descriptive question about a specific food
    or a request for food recommendations.

    Returns:
        str: 'descriptive' or 'recommendation'
    """
    query = query.lower()

    # Descriptive question patterns
    descriptive_patterns = [
        r'apa itu',
        r'jelaskan',
        r'ceritakan',
        r'informasi tentang',
        r'berapa',
        r'apakah',
        r'bagaimana',
        r'seperti apa',
        r'apa saja',
        r'mengapa',
        r'kenapa'
    ]

    # Recommendation request patterns
    recommendation_patterns = [
        r'rekomendasikan',
        r'rekomendasi',
        r'cari',
        r'carikan',
        r'tampilkan',
        r'tunjukkan',
        r'mau makan',
        r'ingin makan',
        r'saya mau',
        r'saya ingin',
        r'saya cari',
        r'kasih saran',
        r'berikan saran',
        r'makanan apa'
    ]

    # Check for descriptive patterns
    for pattern in descriptive_patterns:
        if re.search(pattern, query):
            return 'descriptive'

    # Check for recommendation patterns
    for pattern in recommendation_patterns:
        if re.search(pattern, query):
            return 'recommendation'

    # Default behavior depends on query structure
    # If the query contains food-related filters, it's likely a recommendation request
    filter_patterns = [
        r'sehat',
        r'tidak sehat',
        r'kalori',
        r'makanan berat',
        r'makanan ringan',
        r'cemilan',
        r'minuman',
        r'rendah',
        r'tinggi',
        r'lebih dari',
        r'kurang dari',
        r'antara'
    ]

    for pattern in filter_patterns:
        if re.search(pattern, query):
            return 'recommendation'

    # If query is just a food name or very short, assume it's a descriptive query
    words = query.split()
    if len(words) <= 3:
        return 'descriptive'

    # For ambiguous queries, default to recommendation
    return 'recommendation'

# Ekstrak entitas dari query untuk pemahaman query yang lebih baik
def extract_query_entities(query):
    """
    Ekstrak entitas dari query untuk meningkatkan pemahaman query
    """
    entities = {
        'food_type': None,
        'food_name': None,
        'keterangan_kalori': None,
        'calorie_range': None
    }
    
    query = query.lower()
    
    # Direct food type detection
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
                entities['food_type'] = food_type
                break
        if entities['food_type']:
            break
    
    # If no direct food type is found, try with patterns
    if not entities['food_type']:
        food_patterns = [
            r'(?:makanan|masakan|hidangan|menu)\s+([a-zA-Z\s]+?)\b',  # Match shorter phrases
            r'(?:makanan|masakan|hidangan|menu)\s+([a-zA-Z\s]{2,10})'  # Limit length
        ]
        
        for pattern in food_patterns:
            matches = re.search(pattern, query)
            if matches:
                candidate = matches.group(1).strip()
                # Only accept if it's a reasonable length
                if 1 < len(candidate.split()) < 3 and len(candidate) < 20:
                    entities['food_type'] = candidate
                    break
    
    # Check for invalid food_type values and clean them
    if entities['food_type']:
        # Check if it seems to be a command rather than a food type
        invalid_words = ['berikan', 'cari', 'rekomendasi', 'tampil', 'tunjuk', 'mau', 'ingin', 'saya']
        if any(word in entities['food_type'] for word in invalid_words):
            entities['food_type'] = None
    
    # Kalori keterangan patterns - Improved to be more accurate
    if re.search(r'\brendah\b.*\bkalori\b', query) or re.search(r'\bkalori\b.*\brendah\b', query):
        entities['keterangan_kalori'] = 'rendah'
    elif re.search(r'\bsedang\b.*\bkalori\b', query) or re.search(r'\bkalori\b.*\bsedang\b', query):
        entities['keterangan_kalori'] = 'sedang'
    elif re.search(r'\btinggi\b.*\bkalori\b', query) or re.search(r'\bkalori\b.*\btinggi\b', query):
        entities['keterangan_kalori'] = 'tinggi'
    
    # Ekstrak rentang kalori
    min_cal, max_cal = extract_calorie_filter(query)
    if min_cal is not None or max_cal is not None:
        entities['calorie_range'] = (min_cal, max_cal)

    return entities

# Local console-based interactive QA system to replace Colab widgets
def console_interactive_qa(df, vectorizer, tfidf_matrix):
    """Console-based interactive QA system for local use"""
    print("\n" + "="*60)
    print("🍽️  Sistem Pencarian & Rekomendasi Makanan  🍽️")
    print("="*60)
    print("\n💡 Contoh query:")
    print("  • Apa itu nasi goreng?")
    print("  • Berapa kalori dalam rendang?")
    print("  • Rekomendasi makanan sehat")
    print("  • Cari makanan dengan kalori rendah")
    print("  • Makanan berat dengan kalori di bawah 300")
    print("\nKetik 'exit' atau 'quit' untuk keluar.")
    print("-" * 60)
    
    while True:
        query = input("\n🔍 Masukkan pertanyaan atau permintaan Anda: ")
        
        if not query:
            print("Silakan masukkan pertanyaan atau permintaan.")
            continue
            
        if query.lower() in ['exit', 'quit', 'keluar']:
            print("Terima kasih telah menggunakan sistem QA makanan!")
            break
            
        # Detect query type
        query_type = detect_query_type(query)
        
        if query_type == 'descriptive':
            print(f"\n📝 [Mode Deskriptif] Menjawab pertanyaan: '{query}'")
            print("-" * 60)
            print(answer_query(query, df, vectorizer, tfidf_matrix))
        else:  # recommendation
            print(f"\n🔎 [Mode Rekomendasi] Mencari makanan berdasarkan: '{query}'")
            print("-" * 60)
            results = search_food_with_focal(query, df, vectorizer, tfidf_matrix)
            
            if len(results) > 0:
                print("✨ Hasil pencarian makanan:")
                for i, (_, row) in enumerate(results.iterrows(), 1):
                    print(f"{i}. {row['nama_makanan']} ({row['kalori']} kalori)")
                    print(f"   Jenis: {row['jenis']}")
                    print(f"   Keterangan: {row['keterangan_kalori']} kalori")
                    print(f"   Similarity: {row['similarity']:.4f}")
                    print()
            else:
                print("❌ Tidak ditemukan makanan yang sesuai dengan pencarian Anda.")
                
        print("\n" + "-"*60)

# Replace Colab-specific visualize_food_comparison to be compatible with local environment
def visualize_food_comparison(food_names, df, vectorizer, tfidf_matrix, n=10):
    """
    Visualisasi perbandingan makanan berdasarkan kesamaan
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from wordcloud import WordCloud
        
        # Check if matplotlib is in interactive mode
        if not plt.isinteractive():
            print("Setting matplotlib to interactive mode...")
            plt.ion()
        
        # Ambil subset makanan yang akan dibandingkan
        food_indices = [df[df['nama_makanan'] == name].index[0] for name in food_names if name in df['nama_makanan'].values]

        if len(food_indices) < 2:
            print("Perlu minimal 2 makanan untuk dibandingkan")
            return

        # Hitung similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix[food_indices])

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, cmap='YlGnBu',
                    xticklabels=[df.iloc[idx]['nama_makanan'] for idx in food_indices],
                    yticklabels=[df.iloc[idx]['nama_makanan'] for idx in food_indices])
        plt.title('Similarity antar Makanan')
        plt.tight_layout()
        plt.show()

        # Buat wordcloud untuk setiap makanan
        plt.figure(figsize=(15, len(food_indices)*3))
        for i, idx in enumerate(food_indices):
            food = df.iloc[idx]
            # Dapatkan kata-kata terpenting dari TF-IDF
            feature_names = vectorizer.get_feature_names_out()
            tfidf_vector = tfidf_matrix[idx].toarray()[0]
            word_importance = {feature_names[j]: tfidf_vector[j] for j in range(len(feature_names)) if tfidf_vector[j] > 0}

            # Buat wordcloud
            plt.subplot(len(food_indices), 1, i+1)
            wc = WordCloud(width=800, height=400, background_color='white', max_words=50)
            wc.generate_from_frequencies(word_importance)
            plt.imshow(wc, interpolation='bilinear')
            plt.title(f"Kata Kunci untuk {food['nama_makanan']}")
            plt.axis('off')

        plt.tight_layout()
        plt.show(block=True)
        
    except ImportError as e:
        print(f"Error: Required visualization package missing: {e}")
        print("Please install matplotlib, seaborn, and wordcloud to use visualization features.")

# Main function to run the program locally
def main():
    """Main function to run the program"""
    print("Starting Food Search and QA System...")
    
    # Setup data
    df, vectorizer, tfidf_matrix = setup_data()
    
    # Always use console interface
    console_interactive_qa(df, vectorizer, tfidf_matrix)

###
# Add this to make the script runnable
if __name__ == "__main__":
    main()