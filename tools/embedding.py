import sys

import fastText


def generate_word_frequency_from_embedding(emb_file, out_file):
    emb = fastText.load_model(path=emb_file)
    words, freq = emb.get_words(include_freq=True)
    with open(out_file, 'w+') as f:
        for word, freq in zip(words, freq):
            f.write('{} {}\n'.format(word, freq))


if __name__ == '__main__':
    generate_word_frequency_from_embedding(*sys.argv[1:])
