import pandas as pd
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.complexity.index_category import IndexCategory
from rb.core.lang import Lang
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model

from core.paraphrase.utils import filter_example
from core.utils.rcdoc import create_doc
from core.utils.word_overlap import get_overlap_metrics, get_content_words, get_words, get_edit_distance_metrics, \
    get_pos_ngram_overlap_metrics

DATA_FOLDER = "../../data/"
RESULTS_FOLDER = "../../data/results/"


def get_indexes(lang_model, doc, index_categories):
    cna_graph = CnaGraph(docs=[doc], models=[lang_model])
    try:
        compute_indices(doc=doc, cna_graph=cna_graph)
    except IndexError:
        print("Skip index error")
        print({repr(ind): doc.indices[ind] for ind in doc.indices if ind.category in index_categories})

    local_relevant_indices = {repr(ind): doc.indices[ind] for ind in doc.indices if ind.category in index_categories}

    return local_relevant_indices


def read_and_process_adults():
    doc1 = pd.read_excel(f"{DATA_FOLDER}adults.xlsx")
    doc_mod = doc1[["sourcesentence", "paraphrase_NN",
                    "Garbage", "Irrelevant",
                    "LexicalSimilarity", "SyntacticSimilarity", "SemanticSimilarity",
                    "ParaphraseQuality"
                    ]]
    doc_mod["source"] = doc_mod["sourcesentence"]
    doc_mod["paraphrase"] = doc_mod["paraphrase_NN"]

    return doc_mod


def read_and_process_children():
    doc1 = pd.read_excel(f"{DATA_FOLDER}children.xlsx")
    doc_mod = doc1[["Original Sentence", "Cleaned Submissions",
                    "Garbage", "Irrelevant",
                    "LexicalSimilarity", "SyntacticSimilarity", "SemanticSimilarity",
                    "ParaphraseQuality"
                    ]]
    doc_mod["source"] = doc_mod["Original Sentence"]
    doc_mod["paraphrase"] = doc_mod["Cleaned Submissions"]
    doc_mod["LexicalSimilarity"] = doc_mod["LexicalSimilarity"]

    return doc_mod


def read_and_process_msrp():
    train_lines = open(f"{DATA_FOLDER}msr_paraphrase_train.txt", "r").readlines()
    test_lines = open(f"{DATA_FOLDER}msr_paraphrase_test.txt", "r").readlines()

    results = {}
    for dataset, name in [(train_lines[1:], "train"),
                          (test_lines[1:], "test")]:
        table = [x.split("\t") for x in dataset]
        d = {'ParaphraseQuality': [x[0] for x in table],
             'source': [x[3].strip().replace("\"", "") for x in table],
             'paraphrase': [x[4].strip().replace("\"", "") for x in table],
             }
        df = pd.DataFrame(data=d)
        results[name] = df
    return results


def read_and_process_ULPC():
    doc1 = pd.read_excel(f"{DATA_FOLDER}ULPC.xls")
    doc_mod = doc1[["Index", "Clean_Target_Sentence", "Clean_Utterance",
                    "Garbage_content_bin", "Irrelevant_bin",
                    "Paraphrase_Quality", "Paraphrase_quality_bin", "Paraphrase_quality_tri",
                    "Semantic_Completeness", "Semantic_completeness_bin",
                    "Entailment", "Entailment_bin",
                    "Syntactic_Similarity", "Syntactic_similarity_bin",
                    "Lexical_Similarity", "Lexical_similarity_bin",
                    "Writing_Quality", "Writing_quality_bin",
                    "trn_test_val"
                    ]]

    doc_mod["source"] = doc_mod["Clean_Target_Sentence"]
    doc_mod["paraphrase"] = doc_mod["Clean_Utterance"]
    doc_mod["Garbage"] = doc_mod["Garbage_content_bin"]
    doc_mod["Irrelevant"] = doc_mod["Irrelevant_bin"]
    return doc_mod


def write_dataset_header(f, dataset):
    if dataset == "train_data":
        f.write("Index\tSource\tProduction"
                "\tParaphrase_quality_tri\tParaphrase_quality\tParaphrase_quality_bin"
                "\tSemantic_completeness\tSemantic_completeness_bin"
                "\tEntailment\tEntailment_bin"
                "\tSyntactic_similarity\tSyntactic_similarity_bin"
                "\tLexical_similarity\tLexical_similarity_bin"
                "\tWriting_quality\tWriting_quality_bin"
                "\ttrn_test_val")
    elif dataset == "children" or dataset == "adults":
        f.write("Source\tProduction"
                "\tLexical_similarity_bin"
                "\tSyntactic_similarity_bin"
                "\tSemantic_completeness_bin"
                "\tParaphrase_quality_tri")
    elif dataset == "msrp":
        f.write("Source\tProduction"
                "\tParaphrase_quality_bin"
                "\ttrn_test_val")


def write_dataset_info(f, dataset, doc):
    if dataset == "train_data":
        fields = [  "Paraphrase_quality_tri", "Paraphrase_Quality", "Paraphrase_quality_bin",
                    "Semantic_Completeness", "Semantic_completeness_bin",
                    "Entailment", "Entailment_bin",
                    "Syntactic_Similarity", "Syntactic_similarity_bin",
                    "Lexical_Similarity", "Lexical_similarity_bin",
                    "Writing_Quality", "Writing_quality_bin",
                    "trn_test_val"]
        f.write(f"{doc['Index']}\t{source}\t{prod}\t")
        for field in fields:
            f.write(f"{doc[field]}\t")
    elif dataset == "children":
        fields = ["LexicalSimilarity", "SyntacticSimilarity", "SemanticSimilarity", "ParaphraseQuality"]
        f.write(f"{source}\t{prod}\t")
        for field in fields:
            f.write(f"{doc[field]}\t")
    elif dataset == "adults":
        fields = ["LexicalSimilarity", "SyntacticSimilarity", "SemanticSimilarity", "ParaphraseQuality"]
        f.write(f"{source}\t{prod}\t")
        for field in fields:
            f.write(f"{doc[field]}\t")
    elif dataset == "msrp":
        fields = ["ParaphraseQuality", "trn_test_val"]
        f.write(f"{source}\t{prod}\t")
        for field in fields:
            f.write(f"{doc[field]}\t")


if __name__ == '__main__':
    dataset_options = ["children", "adults", "msrp"]

    for dataset in dataset_options:
        dataset_name = dataset
        if dataset == "train_data":
            doc_mod = read_and_process_ULPC()
        elif dataset == "children":
            doc_mod = read_and_process_children()
        elif dataset == "adults":
            doc_mod = read_and_process_adults()
        elif dataset.startswith("msrp"):
            doc_mod = read_and_process_msrp()
            train_df = doc_mod["train"]
            test_df = doc_mod["test"]
            train_df["trn_test_val"] = 1
            test_df["trn_test_val"] = 3

            doc_mod = pd.concat([train_df, test_df])

        data_dict = []
        print(dataset)
        w2v = create_vector_model(Lang.EN, VectorModelType.from_str("word2vec"), "coca")
        index_categories = [IndexCategory.WORD, IndexCategory.SURFACE, IndexCategory.MORPHOLOGY, IndexCategory.SYNTAX]
        index_categories_combined = [IndexCategory.COHESION]

        indices_list = []
        combined_indices_list = []
        overlap_list = []
        f = open(f"{RESULTS_FOLDER}results_paraphrase_{dataset}.csv", "w")
        i = 0
        for index, data_line in doc_mod.iterrows():
            source = data_line["source"]
            prod = data_line["paraphrase"]

            if dataset == "train_data" and index == 154:
                continue
            if type(source) != str or type(prod)!= str:
                continue
            source_doc = create_doc(source.strip().capitalize())
            prod_doc = create_doc(prod.strip().capitalize())
            if filter_example(source_doc, prod_doc, data_line):
                continue
            print(f"{id} |{source.strip().capitalize()}|-|{prod.strip().capitalize()}|")
            overlap_metrics = get_overlap_metrics(source_doc, prod_doc)
            edit_dist_metrics = get_edit_distance_metrics(source_doc, prod_doc)
            pos_overlap_metrics = get_pos_ngram_overlap_metrics(source_doc, prod_doc)
            source_indexes = get_indexes(w2v, source_doc, index_categories)
            prod_indexes = get_indexes(w2v, prod_doc, index_categories)
            combined_indexes = get_indexes(w2v, prod_doc, index_categories_combined)
            similarity = w2v.similarity(source_doc, prod_doc)

            if len(indices_list) == 0:
                write_dataset_header(f, dataset_name)
                indices_list = list(source_indexes.keys())
                combined_indices_list = list(combined_indexes.keys())
                overlap_list = [key for key in overlap_metrics]
                edit_dist_list = [key for key in edit_dist_metrics]
                pos_overlap_list = [key for key in pos_overlap_metrics]
                indices_list.sort()
                combined_indices_list.sort()
                overlap_list.sort()
                edit_dist_list.sort()
                pos_overlap_list.sort()
                f.write("\t" + "\t".join([f"{i}_source" for i in indices_list]))
                f.write("\t" + "\t".join([f"{i}_prod" for i in indices_list]))
                f.write("\t" + "\t".join([f"{i}_combined" for i in combined_indices_list]))
                f.write("\t" + "\t".join(overlap_list))
                f.write("\t" + "\t".join(edit_dist_list))
                f.write("\t" + "\t".join(pos_overlap_list))
                f.write("\tw2v_similarity\n")
            i += 1
            if i % 250 == 0:
                print(i)
            write_dataset_info(f, dataset_name, data_line)
            f.write("\t".join([str(source_indexes[key]) for key in indices_list]) + "\t")
            f.write("\t".join([str(prod_indexes[key]) for key in indices_list]) + "\t")
            f.write("\t".join([str(combined_indexes[key]) for key in combined_indices_list]) + "\t")
            f.write("\t".join([str(overlap_metrics[key]) for key in overlap_list]) + "\t")
            f.write("\t".join([str(edit_dist_metrics[key]) for key in edit_dist_list]) + "\t")
            f.write("\t".join([str(pos_overlap_metrics[key]) for key in pos_overlap_list]) + "\t")
            f.write(f"{similarity}\n")
        f.close()
