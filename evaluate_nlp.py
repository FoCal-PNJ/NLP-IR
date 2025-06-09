import pandas as pd
import numpy as np
import os
import time
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from information import (setup_data, detect_query_type, extract_query_entities,
                        answer_query, search_food_with_focal)

# Sisa kode tetap sama
def create_evaluation_dataset():
    """
    Buat dataset pengujian untuk evaluasi model
    """
    # Dataset untuk pengujian klasifikasi query
    classification_data = [
        # Query deskriptif
        ("Apa itu nasi goreng?", "descriptive"),
        ("Jelaskan tentang sate padang", "descriptive"),
        ("Apa kandungan gizi dalam rendang?", "descriptive"),
        ("Berapa kalori dalam soto ayam?", "descriptive"),
        ("Apa itu karedok?", "descriptive"),
        ("Bagaimana cara membuat tempe goreng?", "descriptive"),
        ("Apakah wedang jahe sehat?", "descriptive"),
        ("Mengapa es kelapa muda disukai banyak orang?", "descriptive"),
        ("Apa saja bahan dalam gado-gado?", "descriptive"),
        ("Informasi tentang kopi tubruk", "descriptive"),
        
        # Query rekomendasi
        ("Rekomendasikan makanan rendah kalori", "recommendation"),
        ("Cari cemilan sehat", "recommendation"),
        ("Saya ingin minuman dengan kalori sedang", "recommendation"),
        ("Makanan berat dengan kalori rendah", "recommendation"),
        ("Tampilkan minuman tinggi kalori", "recommendation"),
        ("Makanan ringan yang sehat untuk diet", "recommendation"),
        ("Rekomendasi makanan berat enak", "recommendation"),
        ("Berikan contoh cemilan kalori rendah", "recommendation"),
        ("Makanan untuk sarapan", "recommendation"),
        ("Minuman yang bagus untuk kesehatan", "recommendation"),
    ]
    
    # Dataset untuk pengujian retrieval
    retrieval_queries = [
        {
            "query": "Makanan ringan dengan kalori rendah",
            "criteria": {
                "jenis": "makanan ringan",
                "keterangan_kalori": "rendah"
            }
        },
        {
            "query": "Makanan berat yang kalorinya dibawah 300",
            "criteria": {
                "jenis": "makanan berat",
                "max_kalori": 300
            }
        },
        {
            "query": "Minuman kalori tinggi",
            "criteria": {
                "jenis": "minuman",
                "keterangan_kalori": "tinggi"
            }
        },
        {
            "query": "Cemilan dengan kalori sedang",
            "criteria": {
                "jenis": "cemilan",
                "keterangan_kalori": "sedang"
            }
        },
        {
            "query": "Minuman di bawah 100 kalori",
            "criteria": {
                "jenis": "minuman",
                "max_kalori": 100
            }
        }
    ]
    
    # Data set untuk pengujian question answering
    qa_queries = [
        {
            "query": "Apa itu gado-gado?",
            "expected_keywords": ["gado-gado"]
        },
        {
            "query": "Berapa kalori dalam rendang?",
            "expected_keywords": ["rendang"]
        },
        {
            "query": "Jenis makanan apa soto ayam?",
            "expected_keywords": ["soto ayam"]
        },
        {
            "query": "Keterangan kalori pecel lele",
            "expected_keywords": ["pecel lele"]
        }
    ]
    
    return {
        "classification": classification_data,
        "retrieval": retrieval_queries,
        "qa": qa_queries
    }

def evaluate_query_classification(df, vectorizer, tfidf_matrix):
    """
    Evaluasi performa klasifikasi jenis query
    """
    print("\n" + "="*60)
    print(" EVALUASI KLASIFIKASI QUERY")
    print("="*60)
    
    # Ambil dataset pengujian
    test_data = create_evaluation_dataset()["classification"]
    
    y_true = [label for _, label in test_data]
    y_pred = []
    times = []
    
    # Prediksi untuk setiap query
    for query, _ in test_data:
        start_time = time.time()
        pred = detect_query_type(query)
        elapsed_time = time.time() - start_time
        
        y_pred.append(pred)
        times.append(elapsed_time)
    
    # Hitung metrik
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    avg_time = sum(times) / len(times)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=["descriptive", "recommendation"])
    
    # Menampilkan hasil
    print(f"Jumlah query pengujian: {len(test_data)}")
    print(f"\nMetrik Performa:")
    print(f" Presisi: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1-Score: {f1:.4f}")
    print(f" Waktu rata-rata: {avg_time*1000:.2f} ms/query")

    # Visualisasi Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
               xticklabels=["descriptive", "recommendation"],
               yticklabels=["descriptive", "recommendation"])
    plt.title('Confusion Matrix - Klasifikasi Query')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Label Prediksi')
    
    # Simpan hasil
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/confusion_matrix_classification.png")
    plt.close()
    
    # Simpan detail hasil
    results = {
        "accuracy": sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_time_ms": avg_time * 1000,
        "predictions": []
    }
    
    for i, (query, true_label) in enumerate(test_data):
        results["predictions"].append({
            "query": query,
            "true_label": true_label,
            "predicted_label": y_pred[i],
            "correct": true_label == y_pred[i],
            "time_ms": times[i] * 1000
        })
    
    with open(f"{output_dir}/classification_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return precision, recall, f1, avg_time

def evaluate_information_retrieval(df, vectorizer, tfidf_matrix):
    """
    Evaluasi performa pencarian informasi/rekomendasi makanan
    """
    print("\n" + "="*60)
    print(" EVALUASI PENCARIAN MAKANAN")
    print("="*60)
    
    # Ambil dataset pengujian
    retrieval_queries = create_evaluation_dataset()["retrieval"]
    
    total_queries = len(retrieval_queries)
    precision_values = []
    recall_values = []
    f1_values = []
    times = []
    ndcg_values = []
    
    detailed_results = []
    
    for query_data in retrieval_queries:
        query = query_data["query"]
        criteria = query_data["criteria"]
        
        print(f"\nEvaluasi query: '{query}'")
        print(f"Kriteria: {criteria}")
        
        # Jalankan pencarian
        start_time = time.time()
        results = search_food_with_focal(query, df, vectorizer, tfidf_matrix)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        # Periksa hasilnya
        if results.empty or results.iloc[0]["nama_makanan"] in ["Tidak ditemukan", "Error"]:
            print(" Tidak ada hasil ditemukan.")
            precision = 0
            recall = 0
            f1 = 0
            ndcg = 0
        else:
            # Hitung berapa banyak hasil yang sesuai kriteria
            result_count = len(results)
            relevant_count = 0
            relevance_scores = []
            
            for _, row in results.iterrows():
                score = 0  # Skor relevansi (0-3)
                max_possible_score = 0  # Skor maksimum yang mungkin
                
                # Periksa jenis makanan
                if "jenis" in criteria:
                    max_possible_score += 1
                    if row["jenis"].lower() == criteria["jenis"].lower():
                        score += 1
                
                # Periksa keterangan kalori
                if "keterangan_kalori" in criteria:
                    max_possible_score += 1
                    if row["keterangan_kalori"].lower() == criteria["keterangan_kalori"].lower():
                        score += 1
                
                # Periksa batas kalori
                if "max_kalori" in criteria:
                    max_possible_score += 1
                    if row["kalori"] <= criteria["max_kalori"]:
                        score += 1
                if "min_kalori" in criteria:
                    max_possible_score += 1
                    if row["kalori"] >= criteria["min_kalori"]:
                        score += 1
                
                # Hitung skor relevansi dinormalisasi (0-1)
                normalized_score = score / max_possible_score if max_possible_score > 0 else 0
                relevance_scores.append(normalized_score)
                
                # Jika skor relevansi > 0.5, anggap relevan
                if normalized_score >= 0.5:
                    relevant_count += 1
            
            # Hitung precision and recall
            precision = relevant_count / result_count if result_count > 0 else 0
            expected_relevant = df.copy()
            
            # Terapkan semua filter yang ada dalam kriteria
            filtered_df = df.copy()
            if "jenis" in criteria:
                filtered_df = filtered_df[filtered_df["jenis"].str.lower() == criteria["jenis"].lower()]
            if "keterangan_kalori" in criteria:
                filtered_df = filtered_df[filtered_df["keterangan_kalori"].str.lower() == criteria["keterangan_kalori"].lower()]
            if "max_kalori" in criteria:
                filtered_df = filtered_df[filtered_df["kalori"] <= criteria["max_kalori"]]
            if "min_kalori" in criteria:
                filtered_df = filtered_df[filtered_df["kalori"] >= criteria["min_kalori"]]
                
            total_relevant = len(filtered_df)
            recall = relevant_count / total_relevant if total_relevant > 0 else 0
            
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # nDCG calculation (Normalized Discounted Cumulative Gain)
            dcg = 0
            idcg = 0
            ideal_relevance = sorted(relevance_scores, reverse=True)
            
            # Compute DCG
            for i, rel in enumerate(relevance_scores):
                dcg += rel / np.log2(i + 2)  # Log base 2 of position+1
                
            # Compute ideal DCG
            for i, rel in enumerate(ideal_relevance):
                idcg += rel / np.log2(i + 2)
                
            ndcg = dcg / idcg if idcg > 0 else 0
        
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        ndcg_values.append(ndcg)
        
        # Output detail hasil
        print(f"Jumlah hasil: {len(results) if not results.empty else 0}")
        print(f"Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"nDCG: {ndcg:.4f}")
        print(f"Waktu: {elapsed_time*1000:.2f} ms")
        
        # Simpan detail untuk laporan
        result_items = []
        if not results.empty and results.iloc[0]["nama_makanan"] not in ["Tidak ditemukan", "Error"]:
            for _, row in results.iterrows():
                result_items.append({
                    "nama_makanan": row["nama_makanan"],
                    "jenis": row["jenis"],
                    "kalori": float(row["kalori"]),
                    "keterangan_kalori": row["keterangan_kalori"],
                    "similarity": float(row["similarity"])
                })
        
        detailed_results.append({
            "query": query,
            "criteria": criteria,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ndcg": ndcg,
                "time_ms": elapsed_time * 1000
            },
            "results": result_items
        })
    
    # Hitung rata-rata metrik
    avg_precision = sum(precision_values) / total_queries
    avg_recall = sum(recall_values) / total_queries
    avg_f1 = sum(f1_values) / total_queries
    avg_ndcg = sum(ndcg_values) / total_queries
    avg_time = sum(times) / total_queries
    
    # Tampilkan ringkasan hasil
    print("\n" + "-"*60)
    print("RINGKASAN HASIL EVALUASI RETRIEVAL")
    print("-"*60)
    print(f"Jumlah query pengujian: {total_queries}")
    print(f"\nRata-rata metrik:")
    print(f" Precision: {avg_precision:.4f}")
    print(f" Recall: {avg_recall:.4f}")
    print(f" F1-Score: {avg_f1:.4f}")
    print(f" nDCG: {avg_ndcg:.4f}")
    print(f" Waktu rata-rata: {avg_time*1000:.2f} ms/query")

    # Visualisasi hasil
    plt.figure(figsize=(12, 6))
    
    metrics = ['Precision', 'Recall', 'F1', 'nDCG']
    values = [avg_precision, avg_recall, avg_f1, avg_ndcg]
    
    plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
    plt.ylim(0, 1.0)
    plt.title('Metrik Performa Information Retrieval')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tambahkan nilai di atas bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f'{v:.4f}', ha='center')
    
    # Simpan hasil
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/retrieval_metrics.png")
    plt.close()
    
    # Simpan hasil detail
    retrieval_results = {
        "summary": {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_ndcg": avg_ndcg,
            "avg_time_ms": avg_time * 1000
        },
        "queries": detailed_results
    }
    
    with open(f"{output_dir}/retrieval_results.json", "w", encoding='utf-8') as f:
        json.dump(retrieval_results, f, indent=2)
    
    return avg_precision, avg_recall, avg_f1, avg_ndcg, avg_time

def evaluate_qa_accuracy(df, vectorizer, tfidf_matrix):
    """
    Evaluasi akurasi question-answering untuk query deskriptif
    """
    print("\n" + "="*60)
    print("EVALUASI QUESTION ANSWERING")
    print("="*60)
    
    # Ambil dataset pengujian
    qa_queries = create_evaluation_dataset()["qa"]
    
    total_queries = len(qa_queries)
    accuracy_scores = []
    keyword_coverage_scores = []
    times = []
    
    detailed_results = []
    
    for qa_data in qa_queries:
        query = qa_data["query"]
        expected_keywords = [kw.lower() for kw in qa_data["expected_keywords"]]
        
        print(f"\nEvaluasi query: '{query}'")
        print(f"Expected keywords: {expected_keywords}")
        
        # Jalankan QA
        start_time = time.time()
        answer = answer_query(query, df, vectorizer, tfidf_matrix)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        # Periksa jawaban
        if answer and "Maaf, " not in answer:
            # Evaluasi berdasarkan keberadaan keyword
            answer_lower = answer.lower()
            found_keywords = []
            missing_keywords = []
            
            for keyword in expected_keywords:
                if keyword.lower() in answer_lower:
                    found_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)
            
            # Hitung akurasi berdasarkan keyword coverage
            keyword_coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 1.0
            keyword_coverage_scores.append(keyword_coverage)
            
            # Hitung akurasi subjektif (antara 0-1)
            # Berdasarkan apakah jawaban berisi kata kunci yang diharapkan
            subjective_score = keyword_coverage
            accuracy_scores.append(subjective_score)
            
            print(f"Jawaban: {answer}")
            print(f"Kata kunci ditemukan: {found_keywords}")
            print(f"Kata kunci tidak ditemukan: {missing_keywords}")
            print(f"Keyword coverage: {keyword_coverage:.4f}")
            print(f"Skor subjektif: {subjective_score:.4f}")
        else:
            print(f"Tidak ada jawaban valid: {answer}")
            accuracy_scores.append(0)
            keyword_coverage_scores.append(0)
        
        print(f"Waktu: {elapsed_time*1000:.2f} ms")
        
        # Simpan detail hasil
        detailed_results.append({
            "query": query,
            "expected_keywords": expected_keywords,
            "answer": answer,
            "metrics": {
                "keyword_coverage": keyword_coverage_scores[-1],
                "accuracy": accuracy_scores[-1],
                "time_ms": elapsed_time * 1000
            }
        })
    
    # Hitung rata-rata metrik
    avg_accuracy = sum(accuracy_scores) / total_queries
    avg_keyword_coverage = sum(keyword_coverage_scores) / total_queries
    avg_time = sum(times) / total_queries
    
    # Tampilkan ringkasan hasil
    print("\n" + "-"*60)
    print("RINGKASAN HASIL EVALUASI QA")
    print("-"*60)
    print(f"Jumlah query pengujian: {total_queries}")
    print(f"\nRata-rata metrik:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Keyword coverage: {avg_keyword_coverage:.4f}")
    print(f"Waktu rata-rata: {avg_time*1000:.2f} ms/query")

    # Visualisasi hasil per query
    plt.figure(figsize=(12, 6))
    
    queries = [f"Q{i+1}" for i in range(total_queries)]
    
    x = np.arange(len(queries))  # Posisi label
    width = 0.35  # Lebar bar
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, keyword_coverage_scores, width, label='Keyword Coverage')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('QA Performance by Query')
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Simpan hasil
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/qa_metrics.png")
    plt.close()
    
    # Simpan hasil detail
    qa_results = {
        "summary": {
            "avg_accuracy": avg_accuracy,
            "avg_keyword_coverage": avg_keyword_coverage,
            "avg_time_ms": avg_time * 1000
        },
        "queries": detailed_results
    }
    
    with open(f"{output_dir}/qa_results.json", "w", encoding='utf-8') as f:
        json.dump(qa_results, f, indent=2)
    
    return avg_accuracy, avg_keyword_coverage, avg_time

def run_all_evaluations():
    print("Loading data for evaluation...")
    df, vectorizer, tfidf_matrix = setup_data()
    
    # Buat direktori output jika belum ada
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluations with timing
    start_time = time.time()
    
    # Evaluasi klasifikasi query
    class_precision, class_recall, class_f1, class_time = evaluate_query_classification(df, vectorizer, tfidf_matrix)
    
    # Evaluasi information retrieval
    ir_precision, ir_recall, ir_f1, ir_ndcg, ir_time = evaluate_information_retrieval(df, vectorizer, tfidf_matrix)
    
    # Evaluasi question answering
    qa_accuracy, qa_keyword_coverage, qa_time = evaluate_qa_accuracy(df, vectorizer, tfidf_matrix)
    
    # Hitung total waktu
    total_time = time.time() - start_time
    
    # Tampilkan ringkasan umum
    print("\n\n" + "="*60)
    print(" RINGKASAN EVALUASI KESELURUHAN")
    print("="*60)
    
    print("\nKlasifikasi Query:")
    print(f"  F1-Score: {class_f1:.4f}")
    print(f"   Precision: {class_precision:.4f}")
    print(f"   Recall: {class_recall:.4f}")
    
    print("\nInformation Retrieval:")
    print(f"  F1-Score: {ir_f1:.4f}")
    print(f"   Precision: {ir_precision:.4f}")
    print(f"   Recall: {ir_recall:.4f}")
    print(f"   nDCG: {ir_ndcg:.4f}")
    
    print("\nQuestion Answering:")
    print(f"  Accuracy: {qa_accuracy:.4f}")
    print(f"   Keyword Coverage: {qa_keyword_coverage:.4f}")
    
    print(f"\n Total waktu evaluasi: {total_time:.2f} detik")
    
    # Simpan ringkasan ke file
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": total_time,
        "classification": {
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "avg_time_ms": class_time * 1000
        },
        "information_retrieval": {
            "precision": ir_precision,
            "recall": ir_recall,
            "f1": ir_f1,
            "ndcg": ir_ndcg,
            "avg_time_ms": ir_time * 1000
        },
        "question_answering": {
            "accuracy": qa_accuracy,
            "keyword_coverage": qa_keyword_coverage,
            "avg_time_ms": qa_time * 1000
        }
    }
    
    with open(f"{output_dir}/evaluation_summary.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Buat visualisasi ringkasan
    plt.figure(figsize=(14, 8))
    
    # Buat data untuk plot
    categories = ['Classification', 'Information Retrieval', 'Question Answering']
    
    # Metrik untuk setiap kategori
    metrics = {}
    metrics['Classification'] = [class_precision, class_recall, class_f1]
    metrics['Information Retrieval'] = [ir_precision, ir_recall, ir_f1, ir_ndcg]
    metrics['Question Answering'] = [qa_accuracy, qa_keyword_coverage]
    
    # Labels untuk setiap kategori
    labels = {}
    labels['Classification'] = ['Precision', 'Recall', 'F1']
    labels['Information Retrieval'] = ['Precision', 'Recall', 'F1', 'nDCG']
    labels['Question Answering'] = ['Accuracy', 'Keyword Coverage']
    
    # Buat subplot untuk setiap kategori
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, category in enumerate(categories):
        axs[i].bar(labels[category], metrics[category], color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(metrics[category])])
        axs[i].set_title(category)
        axs[i].set_ylim(0, 1.0)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tampilkan nilai di atas bar
        for j, v in enumerate(metrics[category]):
            axs[i].text(j, v + 0.05, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_summary.png")
    plt.close()
    
    print(f"\nHasil evaluasi telah disimpan di folder '{output_dir}'")

if __name__ == "__main__":
    run_all_evaluations()