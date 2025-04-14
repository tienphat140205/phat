import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Hàm hỗ trợ chia văn bản thành từ
def split_text_word(texts):
    """
    Chia một danh sách chuỗi thành danh sách các danh sách từ.
    
    Args:
        texts (list): Danh sách các chuỗi văn bản.
    
    Returns:
        list: Danh sách các danh sách từ.
    """
    if isinstance(texts, str):
        return [texts.split()]
    return [text.split() for text in texts]


class CrossLingualTopicEvaluator:
    """Class để đánh giá mô hình chủ đề đa ngôn ngữ với TC, TD, TU và cross-lingual alignment."""

    def __init__(self, reference_corpus_lang1=None, reference_corpus_lang2=None, 
                 vocab_lang1=None, vocab_lang2=None, cv_type='c_v'):
        """
        Khởi tạo evaluator với các tham số cần thiết.
        
        Args:
            reference_corpus_lang1 (list): Corpus tham chiếu cho ngôn ngữ 1 (ví dụ: tiếng Anh).
            reference_corpus_lang2 (list): Corpus tham chiếu cho ngôn ngữ 2 (ví dụ: tiếng Trung).
            vocab_lang1 (list): Từ vựng cho ngôn ngữ 1.
            vocab_lang2 (list): Từ vựng cho ngôn ngữ 2.
            cv_type (str): Loại coherence score ('c_v', 'u_mass', 'c_npmi', v.v.). Mặc định: 'c_v'.
        """
        self.reference_corpus_lang1 = reference_corpus_lang1
        self.reference_corpus_lang2 = reference_corpus_lang2
        self.vocab_lang1 = vocab_lang1
        self.vocab_lang2 = vocab_lang2
        self.cv_type = cv_type
        self.dictionary_lang1 = Dictionary(split_text_word(vocab_lang1)) if vocab_lang1 else None
        self.dictionary_lang2 = Dictionary(split_text_word(vocab_lang2)) if vocab_lang2 else None

    def compute_topic_coherence(self, top_words, lang='lang1'):
        """
        Tính Topic Coherence (TC) cho danh sách từ hàng đầu của các chủ đề.
        
        Args:
            top_words (list): Danh sách các chuỗi từ hàng đầu (mỗi chuỗi là một chủ đề).
            lang (str): Ngôn ngữ để tính TC ('lang1' hoặc 'lang2').
        
        Returns:
            tuple: (coherence_per_topic, mean_coherence).
        """
        if lang == 'lang1' and self.reference_corpus_lang1 and self.vocab_lang1:
            reference_corpus = self.reference_corpus_lang1
            dictionary = self.dictionary_lang1
        elif lang == 'lang2' and self.reference_corpus_lang2 and self.vocab_lang2:
            reference_corpus = self.reference_corpus_lang2
            dictionary = self.dictionary_lang2
        else:
            raise ValueError(f"Reference corpus hoặc vocab không được cung cấp cho {lang}")

        split_top_words = split_text_word(top_words)
        num_top_words = len(split_top_words[0])
        for item in split_top_words:
            assert num_top_words == len(item), "Số từ mỗi chủ đề không đồng đều"

        split_reference_corpus = split_text_word(reference_corpus)
        cm = CoherenceModel(
            texts=split_reference_corpus,
            dictionary=dictionary,
            topics=split_top_words,
            topn=num_top_words,
            coherence=self.cv_type
        )
        cv_per_topic = cm.get_coherence_per_topic()
        score = np.mean(cv_per_topic)
        return cv_per_topic, score

    def compute_topic_diversity(self, top_words):
        """
        Tính Topic Diversity (TD) cho danh sách từ hàng đầu của các chủ đề.
        
        Args:
            top_words (list): Danh sách các chuỗi từ hàng đầu (mỗi chuỗi là một chủ đề).
        
        Returns:
            float: Giá trị TD.
        """
        K = len(top_words)
        T = len(top_words[0].split())
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        counter = vectorizer.fit_transform(top_words).toarray()

        TF = counter.sum(axis=0)
        TD = (TF == 1).sum() / (K * T)
        return TD

    def compute_topic_uniqueness(self, top_words):
        """
        Tính Topic Uniqueness (TU) cho danh sách từ hàng đầu của các chủ đề.
        
        Args:
            top_words (list): Danh sách các chuỗi từ hàng đầu (mỗi chuỗi là một chủ đề).
        
        Returns:
            float: Giá trị TU.
        """
        K = len(top_words)
        T = len(top_words[0].split())
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        counter = vectorizer.fit_transform(top_words).toarray()

        TU = 0.0
        TF = counter.sum(axis=0)
        cnt = TF * (counter > 0)

        for i in range(K):
            TU += (1 / cnt[i][np.where(cnt[i] > 0)]).sum() / T
        TU /= K
        return TU

    def compute_cross_lingual_alignment(self, beta_lang1, beta_lang2):
        """
        Tính độ căn chỉnh đa ngôn ngữ bằng cosine similarity giữa các chủ đề của hai ngôn ngữ.
        
        Args:
            beta_lang1 (np.ndarray): Ma trận từ-chủ đề cho ngôn ngữ 1 (shape: [num_topics, vocab_size]).
            beta_lang2 (np.ndarray): Ma trận từ-chủ đề cho ngôn ngữ 2 (shape: [num_topics, vocab_size]).
        
        Returns:
            tuple: (alignment_per_topic, mean_alignment).
        """
        if beta_lang1.shape[0] != beta_lang2.shape[0]:
            raise ValueError("Số lượng chủ đề giữa hai ngôn ngữ không khớp")

        alignment_scores = cosine_similarity(beta_lang1, beta_lang2)
        alignment_per_topic = alignment_scores.diagonal()
        mean_alignment = np.mean(alignment_per_topic)
        return alignment_per_topic, mean_alignment

    def evaluate(self, top_words_lang1, top_words_lang2, beta_lang1=None, beta_lang2=None):
        """
        Đánh giá mô hình chủ đề đa ngôn ngữ với TC, TD, TU và cross-lingual alignment.
        
        Args:
            top_words_lang1 (list): Danh sách từ hàng đầu cho ngôn ngữ 1.
            top_words_lang2 (list): Danh sách từ hàng đầu cho ngôn ngữ 2.
            beta_lang1 (np.ndarray, optional): Ma trận từ-chủ đề cho ngôn ngữ 1.
            beta_lang2 (np.ndarray, optional): Ma trận từ-chủ đề cho ngôn ngữ 2.
        
        Returns:
            dict: Kết quả đánh giá với các chỉ số TC, TD, TU và alignment (nếu có).
        """
        results = {}

        # Tính Topic Coherence
        print("Đang tính Topic Coherence...")
        try:
            tc_lang1_per_topic, tc_lang1_score = self.compute_topic_coherence(top_words_lang1, lang='lang1')
            results['TC_lang1_per_topic'] = tc_lang1_per_topic
            results['TC_lang1_score'] = tc_lang1_score
        except Exception as e:
            print(f"Lỗi khi tính TC cho lang1: {e}")
        
        try:
            tc_lang2_per_topic, tc_lang2_score = self.compute_topic_coherence(top_words_lang2, lang='lang2')
            results['TC_lang2_per_topic'] = tc_lang2_per_topic
            results['TC_lang2_score'] = tc_lang2_score
        except Exception as e:
            print(f"Lỗi khi tính TC cho lang2: {e}")

        # Tính Topic Diversity
        print("Đang tính Topic Diversity...")
        results['TD_lang1'] = self.compute_topic_diversity(top_words_lang1)
        results['TD_lang2'] = self.compute_topic_diversity(top_words_lang2)

        # Tính Topic Uniqueness
        print("Đang tính Topic Uniqueness...")
        results['TU_lang1'] = self.compute_topic_uniqueness(top_words_lang1)
        results['TU_lang2'] = self.compute_topic_uniqueness(top_words_lang2)

        # Tính Cross-lingual Alignment (nếu có beta)
        if beta_lang1 is not None and beta_lang2 is not None:
            print("Đang tính Cross-lingual Alignment...")
            alignment_per_topic, mean_alignment = self.compute_cross_lingual_alignment(beta_lang1, beta_lang2)
            results['alignment_per_topic'] = alignment_per_topic
            results['alignment_score'] = mean_alignment

        return results