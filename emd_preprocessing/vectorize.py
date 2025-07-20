import MeCab
import spacy
import numpy as np

class JaSentenceVectorizer:
    """
    日本語文を MeCab で分かち書き・品詞フィルタリングし、
    ベクトル辞書に基づいて単語ベクトル列に変換するクラス。

    Vectorizes a Japanese sentence using MeCab and a word embedding dictionary.
    """

    def __init__(self, vectors: dict, stop_pos=None):
        """
        Parameters:
        ----------
        vectors : dict
            単語 → ベクトルの辞書 / Word → vector mapping (e.g., fastText, Word2Vec)
        stop_pos : set[str], optional
            除外する品詞タグの集合（例: 助詞, 記号）/ POS tags to ignore
        """
        self.vectors = vectors
        self.mecab = MeCab.Tagger()
        self.stop_pos = stop_pos or {
            '助詞', '助動詞', '記号', '接続詞', '連体詞', '感動詞', 'フィラー'
        }

    def sentence_to_vector(self, sentence: str):
        """
        文を単語に分割し、辞書に存在する単語ベクトルを抽出する。

        Parameters:
        ----------
        sentence : str
            日本語の1文

        Returns:
        -------
        words : list[str]
            使用された単語リスト
        vectors : np.ndarray of shape (n_words, dim)
            対応する単語ベクトル配列
        """
        words, vecs = [], []
        node = self.mecab.parseToNode(sentence)
        while node:
            word = node.surface
            pos = node.feature.split(',')[0]
            if word in self.vectors and pos not in self.stop_pos:
                words.append(word)
                vecs.append(self.vectors[word])
            node = node.next
        return words, np.array(vecs)


class EnSentenceVectorizer:
    """
    英語文を spaCy でトークン化・品詞フィルタリングし、
    ベクトル辞書に基づいて単語ベクトル列に変換するクラス。

    Vectorizes an English sentence using spaCy and a word embedding dictionary.
    """

    def __init__(self, vectors: dict, stop_pos=None):
        """
        Parameters:
        ----------
        vectors : dict
            単語 → ベクトルの辞書 / Word → vector mapping
        stop_pos : set[str], optional
            除外する品詞タグ（spaCyのPOS表記）/ POS tags to ignore
        """
        self.vectors = vectors
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_pos = stop_pos or {
            'DET', 'ADP', 'CCONJ', 'PART', 'INTJ', 'PUNCT', 'SYM'
        }

    def sentence_to_vector(self, sentence: str):
        """
        英文を単語に分割し、辞書に存在する単語ベクトルを抽出する。

        Parameters:
        ----------
        sentence : str
            英語の1文

        Returns:
        -------
        words : list[str]
            使用された単語リスト
        vectors : np.ndarray of shape (n_words, dim)
            対応する単語ベクトル配列
        """
        words, vecs = [], []
        doc = self.nlp(sentence)
        for token in doc:
            if token.text in self.vectors and token.pos_ not in self.stop_pos:
                words.append(token.text)
                vecs.append(self.vectors[token.text])
        return words, np.array(vecs)
