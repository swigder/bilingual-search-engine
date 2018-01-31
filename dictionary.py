import argparse

from gensim.models.keyedvectors import EuclideanKeyedVectors


class Dictionary:
    def __init__(self, src_emb_file, tgt_emb_file):
        self.src_emb = EuclideanKeyedVectors.load_word2vec_format(src_emb_file, binary=False)
        self.tgt_emb = EuclideanKeyedVectors.load_word2vec_format(tgt_emb_file, binary=False)

    def translate(self, src_word, topn=1):
        src_vector = self.src_emb.word_vec(src_word)
        return self.tgt_emb.similar_by_vector(src_vector, topn=topn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Word Segmenter.')

    parser.add_argument('src_emb_file', type=str, help='File with source embeddings')
    parser.add_argument('tgt_emb_file', type=str, help='File with target embeddings')
    parser.add_argument('-n', '--top_n', type=int, default=1, help='Number of translations to provide')
    # parser.add_argument('--mode', default='command', choices=['command', 'interactive'])
    # parser.add_argument()

    args = parser.parse_args()

    dict = Dictionary(args.src_emb_file, args.tgt_emb_file)

    while True:
        word = input(">> ")
        print(dict.translate(word, topn=args.top_n))
