from flask import Flask, render_template, request, jsonify, send_file, session, Response
from youtube_scraper import YouTubeScraper
from sentiment_analyzer import SentimentAnalyzer
from data_storage import DataStorage
import os
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO
import json
import base64
import time
from queue import Queue
import threading

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here-change-this-in-production')
app.config['SESSION_TYPE'] = 'filesystem'

API_KEY = os.getenv('YOUTUBE_API_KEY')
scraper = YouTubeScraper(API_KEY)
analyzer = SentimentAnalyzer()
storage = DataStorage()

# Store data in memory (in production, use database)
app_data = {
    'scraped_data': None,
    'analysis_results': None
}

# Progress tracking
progress_data = {
    'current': 0,
    'total': 0,
    'status': 'idle',
    'message': ''
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/comments')
def comments():
    return render_template('comments.html')

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/scrape', methods=['POST'])
def scrape_comments():
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        max_comments = int(data.get('max_comments', 100))
        include_replies = data.get('include_replies', False)
        
        # Limit to 100000
        max_comments = min(max_comments, 100000)
        
        if not video_url:
            return jsonify({'error': 'URL video tidak boleh kosong'}), 400
        
        # Extract video ID
        video_id = scraper.extract_video_id(video_url)
        if not video_id:
            return jsonify({'error': 'URL video tidak valid'}), 400
        
        # Get video info
        video_info = scraper.get_video_info(video_id)
        
        # Check available comments in video
        available_comments = int(video_info.get('comments', 0)) if video_info else 0
        
        # Reset progress
        progress_data['current'] = 0
        progress_data['total'] = max_comments
        progress_data['status'] = 'scraping'
        progress_data['message'] = 'Memulai scraping...'
        progress_data['available'] = available_comments
        
        # Progress callback
        def update_progress(current, total):
            progress_data['current'] = current
            progress_data['total'] = total
            progress_data['available'] = available_comments
            progress_data['message'] = f'Mengambil komentar {current}/{total}...'
        
        # Scrape comments (with or without replies)
        comments_data = scraper.get_comments(
            video_id, 
            max_results=max_comments, 
            include_replies=include_replies,
            progress_callback=update_progress
        )
        
        if not comments_data:
            progress_data['status'] = 'error'
            progress_data['message'] = 'Tidak ada komentar ditemukan'
            return jsonify({'error': 'Tidak ada komentar ditemukan'}), 404
        
        # Check if we got less than requested
        got_less = len(comments_data) < max_comments
        warning_message = None
        if got_less and available_comments > 0:
            warning_message = f'Video ini hanya memiliki {len(comments_data)} komentar dari {max_comments} yang diminta. Semua komentar yang tersedia sudah diambil.'
        
        # Save to file
        filename = storage.save_comments(video_id, video_info, comments_data)
        
        # Store in app_data
        app_data['scraped_data'] = {
            'video_id': video_id,
            'video_url': video_url,
            'video_info': video_info,
            'comments': comments_data,
            'total_comments': len(comments_data),
            'saved_filename': filename,
            'include_replies': include_replies,
            'requested_comments': max_comments,
            'available_comments': available_comments
        }
        app_data['analysis_results'] = None  # Reset analysis
        
        # Update progress
        progress_data['status'] = 'completed'
        progress_data['current'] = len(comments_data)
        progress_data['message'] = f'Selesai! {len(comments_data)} komentar berhasil diambil'
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'video_info': video_info,
            'total_comments': len(comments_data),
            'requested_comments': max_comments,
            'available_comments': available_comments,
            'comments': comments_data,
            'saved_filename': filename,
            'include_replies': include_replies,
            'warning': warning_message
        })
        
    except Exception as e:
        progress_data['status'] = 'error'
        progress_data['message'] = f'Error: {str(e)}'
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def get_progress():
    """Endpoint to get scraping progress"""
    return jsonify(progress_data)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get comments from request or app_data
        request_data = request.json or {}
        comments = request_data.get('comments')
        
        # If no comments in request, try to get from scraped_data
        if not comments:
            if 'scraped_data' in app_data and app_data['scraped_data']:
                comments = app_data['scraped_data'].get('comments')
        
        if not comments:
            return jsonify({'error': 'Tidak ada komentar untuk dianalisis'}), 400
        
        # Get selected methods (updated with new ML methods)
        selected_methods = request_data.get('methods', ['naive_bayes', 'svm', 'lstm', 'indobert'])
        
        # Analyze sentiment (FAST - no wordcloud yet)
        results = analyzer.analyze_multiple_methods(comments, selected_methods)
        
        # Generate confusion matrix images if multiple methods
        confusion_matrix_images = {}
        if len(selected_methods) > 1 and 'confusion_matrices' in results:
            confusion_matrix_images = analyzer.generate_confusion_matrix_images(results['confusion_matrices'])
        
        # Store results (wordcloud will be generated on demand)
        app_data['analysis_results'] = results
        app_data['wordclouds'] = None  # Lazy loading
        app_data['confusion_matrix_images'] = confusion_matrix_images
        
        response = {
            'success': True,
            'summary': results['summary'],
            'preprocessing_examples': results.get('preprocessing_examples', [])
        }
        
        # Add accuracy if available
        if 'accuracy' in results:
            response['accuracy'] = results['accuracy']
        
        # Add confusion matrices if available
        if confusion_matrix_images:
            response['confusion_matrices'] = confusion_matrix_images
        
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Generate wordcloud on demand (lazy loading)"""
    try:
        if 'analysis_results' not in app_data or not app_data['analysis_results']:
            return jsonify({'error': 'Belum ada hasil analisis'}), 400
        
        # Generate wordcloud only when requested
        if app_data.get('wordclouds') is None:
            results = app_data['analysis_results']
            wordclouds = analyzer.generate_wordcloud(results['comments'])
            app_data['wordclouds'] = wordclouds
        
        wordclouds = app_data['wordclouds']
        
        return jsonify({
            'success': True,
            'wordcloud_positive': wordclouds.get('positive'),
            'wordcloud_negative': wordclouds.get('negative'),
            'positive_count': wordclouds.get('positive_count', 0),
            'negative_count': wordclouds.get('negative_count', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/saved_files', methods=['GET'])
def get_saved_files():
    try:
        files = storage.get_saved_files()
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        print(f"Error getting saved files: {e}")
        return jsonify({'success': True, 'files': []})

@app.route('/load_file/<filename>', methods=['GET'])
def load_file(filename):
    data = storage.load_comments(filename)
    if data:
        # Store in app_data
        app_data['scraped_data'] = {
            'video_id': data['video_id'],
            'video_info': data['video_info'],
            'comments': data['comments'],
            'total_comments': data['total_comments'],
            'saved_filename': filename
        }
        return jsonify({'success': True, 'data': data})
    return jsonify({'success': False, 'error': 'File tidak ditemukan'}), 404

@app.route('/delete_file/<filename>', methods=['DELETE'])
def delete_file(filename):
    if storage.delete_file(filename):
        return jsonify({'success': True, 'message': 'File berhasil dihapus'})
    return jsonify({'success': False, 'error': 'File tidak ditemukan'}), 404

@app.route('/get_scraped_data', methods=['GET'])
def get_scraped_data():
    if app_data['scraped_data']:
        return jsonify({
            'success': True,
            'data': app_data['scraped_data']
        })
    return jsonify({'success': False, 'error': 'Belum ada data scraping'}), 404

@app.route('/get_analysis_results', methods=['GET'])
def get_analysis_results():
    if app_data['analysis_results']:
        return jsonify({
            'success': True,
            'results': app_data['analysis_results'],
            'video_info': app_data['scraped_data']['video_info'] if app_data['scraped_data'] else None
        })
    return jsonify({'success': False, 'error': 'Belum ada hasil analisis'}), 404

@app.route('/export', methods=['GET'])
def export_data():
    try:
        if not app_data['analysis_results']:
            return jsonify({'error': 'Tidak ada data untuk di-export'}), 400
        
        results = app_data['analysis_results']
        video_info = app_data['scraped_data']['video_info'] if app_data['scraped_data'] else None
        
        # Create Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Video Info
            if video_info:
                df_info = pd.DataFrame([video_info])
                df_info.to_excel(writer, sheet_name='Info Video', index=False)
            
            # Sheet 2: Comments with sentiment
            df_comments = pd.DataFrame(results['comments'])
            df_comments.to_excel(writer, sheet_name='Komentar & Sentimen', index=False)
            
            # Sheet 3: Summary statistics
            summary_data = []
            for method, stats in results['summary'].items():
                summary_data.append({
                    'Metode': method,
                    'Positif': stats['positive'],
                    'Negatif': stats['negative'],
                    'Netral': stats['neutral'],
                    'Total': stats['positive'] + stats['negative'] + stats['neutral'],
                    'Persentase Positif': f"{stats['positive']/(stats['positive']+stats['negative']+stats['neutral'])*100:.1f}%",
                    'Persentase Negatif': f"{stats['negative']/(stats['positive']+stats['negative']+stats['neutral'])*100:.1f}%",
                    'Persentase Netral': f"{stats['neutral']/(stats['positive']+stats['negative']+stats['neutral'])*100:.1f}%"
                })
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Ringkasan Metode', index=False)
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='analisis_sentimen_youtube.xlsx'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    """Analyze single comment input from user"""
    try:
        data = request.get_json()
        comment_text = data.get('comment', '').strip()
        
        if not comment_text:
            return jsonify({'error': 'Komentar tidak boleh kosong'}), 400
        
        # Preprocess
        preprocessed = analyzer.preprocessor.preprocess_simple(comment_text)
        
        # Analyze with all 4 methods
        results = {
            'original': comment_text,
            'preprocessed': preprocessed,
            'sentiments': {}
        }
        
        # Naive Bayes
        nb_sentiment, nb_score = analyzer.naive_bayes_analysis(preprocessed)
        results['sentiments']['naive_bayes'] = {
            'label': nb_sentiment,
            'score': nb_score,
            'name': 'Naive Bayes'
        }
        
        # SVM
        svm_sentiment, svm_score = analyzer.svm_analysis(preprocessed)
        results['sentiments']['svm'] = {
            'label': svm_sentiment,
            'score': svm_score,
            'name': 'SVM'
        }
        
        # LSTM
        lstm_sentiment, lstm_score = analyzer.lstm_analysis(preprocessed)
        results['sentiments']['lstm'] = {
            'label': lstm_sentiment,
            'score': lstm_score,
            'name': 'LSTM'
        }
        
        # IndoBERT
        indobert_sentiment, indobert_score = analyzer.indobert_analysis(preprocessed)
        results['sentiments']['indobert'] = {
            'label': indobert_sentiment,
            'score': indobert_score,
            'name': 'IndoBERT'
        }
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/preprocess_text', methods=['POST'])
def preprocess_text():
    """Endpoint untuk mendapatkan detail preprocessing steps untuk satu text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        
        # Menggunakan text_preprocessor untuk mendapatkan detail steps
        from text_preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        # Get detailed steps
        steps = {}
        
        # Original
        steps['original'] = text
        steps['original_words'] = len(text.split())
        
        # Case folding
        case_folded = preprocessor.case_folding(text)
        steps['case_folding'] = case_folded
        
        # Noise removal (URL, mentions, hashtags, numbers)
        noise_removed = preprocessor.remove_noise(case_folded)
        steps['noise_removal'] = noise_removed
        
        # Remove numbers (already in remove_noise)
        steps['remove_numbers'] = noise_removed
        
        # Remove punctuation
        no_punct = preprocessor.remove_punctuation(noise_removed)
        steps['remove_punctuation'] = no_punct
        
        # Spelling correction (normalization)
        corrected = preprocessor.normalize_text(no_punct)
        steps['spelling_correction'] = corrected
        
        # Tokenization
        tokens = preprocessor.tokenize(corrected)
        steps['tokenization'] = tokens
        steps['token_count'] = len(tokens)
        
        # Normalization (already done above in spelling_correction)
        steps['normalization'] = tokens
        
        # Stopword removal
        no_stopwords = preprocessor.remove_stopwords(tokens)
        steps['stopword_removal'] = no_stopwords
        steps['tokens_after_stopword'] = len(no_stopwords)
        
        # Stemming
        stemmed = preprocessor.stem_text(no_stopwords)
        steps['stemming'] = stemmed
        
        # Negation handling
        negation_handled = preprocessor.handle_negation(stemmed)
        steps['negation_handling'] = negation_handled
        
        # Final result
        final = ' '.join(negation_handled) if isinstance(negation_handled, list) else negation_handled
        steps['final'] = final
        steps['final_words'] = len(negation_handled) if isinstance(negation_handled, list) else len(final.split())
        
        # Calculate reduction rate
        if steps['original_words'] > 0:
            reduction = ((steps['original_words'] - steps['final_words']) / steps['original_words']) * 100
            steps['reduction_rate'] = round(reduction, 1)
        else:
            steps['reduction_rate'] = 0
        
        return jsonify({
            'success': True,
            'steps': steps
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
