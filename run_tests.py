import argparse
import os

from collections import namedtuple

from dictionary import MonolingualDictionary
from ir_data_reader import readers, sub_collection
from search_engine import EmbeddingSearchEngine
from baseline import CosineSimilaritySearchEngine


PrecisionRecall = namedtuple('PrecisionRecall', ['precision', 'recall'])


def precision_recall(expected, actual):
    true_positives = sum([1 for item in expected if item in actual])
    return true_positives / len(actual), true_positives / len(expected)


def read_data(data_dir, doc_file, query_file, relevance_file, reader):
    f = lambda file: os.path.join(data_dir, file)

    return reader.read(f(doc_file), f(query_file), f(relevance_file))


def query_result(search_engine, i, query, expected, documents, n=5, verbose=False):
    results = search_engine.query_index(query, n_results=n)
    results_i = []
    if verbose:
        print()
        print(i, query, expected)
    for distance, result in results:
        result_i = documents[result]
        results_i.append(result_i)
        if verbose:
            print(result_i, distance, result[:300])
            if result_i in expected:
                print('Correct!')
    precision, recall = precision_recall(expected=expected, actual=results_i)
    return PrecisionRecall(precision, recall)


def compare_search_engines(search_engine, baseline, collection, n=5):
    total_precision, total_recall = 0, 0
    base_precision, base_recall = 0, 0
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}

    for i, query in collection.queries.items():
        expected = collection.relevance[i]

        engine_pr = query_result(search_engine, i, query, expected, doc_ids, n, verbose=False)
        base_pr = query_result(baseline, i, query, expected, doc_ids, n, verbose=False)

        total_precision += engine_pr.precision
        total_recall += engine_pr.recall
        base_precision += base_pr.precision
        base_recall += base_pr.recall

        if engine_pr.precision < base_pr.precision or engine_pr.recall < base_pr.recall:
            query_result(search_engine, i, query, expected, doc_ids, n, verbose=True)
            print('System: {:.4f} / {:.4f}'.format(engine_pr.precision, engine_pr.recall))
            query_result(baseline, i, query, expected, doc_ids, n, verbose=True)
            print('Baseline: {:.4f} / {:.4f}'.format(base_pr.precision, base_pr.recall))

    print()
    print('         Precision / Recall: {:.4f} / {:.4f}'.format(total_precision / len(collection.queries),
                                                                total_recall / len(collection.queries)))
    print('Baseline Precision / Recall: {:.4f} / {:.4f}'.format(base_precision / len(collection.queries),
                                                                base_recall / len(collection.queries)))


def test_search_engine(search_engine, collection, n=5, verbose=False):
    total_precision, total_recall = 0, 0
    doc_ids = {doc_text: doc_id for doc_id, doc_text in collection.documents.items()}
    for i, query in collection.queries.items():
        expected = collection.relevance[i]
        pr = query_result(search_engine, i, query, expected, doc_ids, n, verbose=True)
        total_precision += pr.precision
        total_recall += pr.recall
    if verbose:
        print()
    print('Precision / Recall: {:.4f} / {:.4f}'.format(total_precision / len(collection.queries),
                                                       total_recall / len(collection.queries)))


def sub_query(query_i, n=None, search_engine=None):
    n = n if n else args.number_results
    search_engine = search_engine if search_engine else list(search_engines.values())[0]
    query_collection = sub_collection(ir_collection, query_i)
    test_search_engine(search_engine, query_collection, n=n, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IR data reader.')

    parser.add_argument('ir_dir', type=str, help='Directory with IR files', nargs='?')
    parser.add_argument('embed', type=str, help='Embedding file', nargs='?')
    parser.add_argument('-t', '--type', choices=readers.keys(), default='time')
    parser.add_argument('-n', '--number_results', type=int, default=5)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-b', '--baseline', action='store_true')
    parser.add_argument('-c', '--compare', action='store_true')

    args = parser.parse_args()

    if not args.ir_dir:
        args.ir_dir = '/Users/xx/Documents/school/kth/thesis/ir-datasets/'
    if not args.embed:
        args.embed = '/Users/xx/Downloads/MUSE-master/trained/vectors-en.txt'

    reader = readers[args.type](os.path.join(args.ir_dir, args.type))
    ir_collection = reader.read_documents_queries_relevance()

    search_engines = {}

    if args.compare or not args.baseline:
        mono_dict = MonolingualDictionary(emb_file=args.embed)
        search_engines['Embedding'] = EmbeddingSearchEngine(dictionary=mono_dict)
    if args.compare or args.baseline:
        search_engines['Baseline'] = CosineSimilaritySearchEngine()

    for engine in search_engines.values():
        engine.index_documents(ir_collection.documents.values())

    if args.compare:
        compare_search_engines(search_engines['Embedding'], search_engines['Baseline'],
                               ir_collection, n=args.number_results)
    else:
        engine = list(search_engines.items())[0]
        test_search_engine(engine, ir_collection, n=args.number_results, verbose=args.verbose)
