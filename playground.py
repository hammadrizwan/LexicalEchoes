from xml.parsers.expat import model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
# from transformers.utils import default_cache_path
import nethook
import os,json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import linecache
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from model_files.qwen import qwen_test_scpp
from model_files.e5_functions import e5_test_scpp,average_pool
from model_files.qwen import qwen_get_embeddings

def load_model(model_name="meta-llama/Meta-Llama-3-8B",access_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use float16 if needed
        device_map="auto",
        token=access_token,  # Use access token if required
    )
    model.eval()
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def e5(device):
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct').to(device)
    input_texts=[]
    # input_texts.append("The city park comes alive in the early morning, with joggers tracing the winding paths and birds greeting the rising sun. The cool air carries the scent of freshly cut grass, while the soft light filters through the leaves. It’s a peaceful scene, one that feels both vibrant and calm at the same time.")
    # input_texts.append("At dawn, the park stirs with quiet activity as runners follow the curving trails and sparrows call to one another. A faint breeze spreads the aroma of mown grass, and sunlight dapples the ground through the trees. The moment feels serene yet full of life, a gentle harmony between motion and stillness.")
    # input_texts.append("The city park comes alive in the early morning, not with joggers or birdsong, but with the sounds of construction crews preparing for a major renovation. The cool air carries the scent of fresh paint and sawdust, while the soft light filters through the leaves.  It’s a peaceful scene, one that feels both chaotic and calm at the same time.")
    
    input_texts=["The finalists will play in Giants stadium.",
                 "Giants stadium will be the playground for the finalists.",
                 "North stadium will be the playground for the finalists."]
    # input_texts=[
    #     "Can I get almond milk from the nearest supermarket, and if so, is it available in larger cartons suitable for cooking and baking purposes?",
    #     "Does the supermarket close by stock almond milk, and do they offer multiple brands or sizes, especially larger ones for frequent use?",
    #     "Can I get oat milk from the nearest supermarket, and if so, is it available in larger cartons suitable for cooking and baking purposes?"
    # ]
#     input_texts=[
#     "The researcher analyzes the data.",
#     "The researcher is analyzing the data.",
#     "The researcher analyzed the data yesterday."
# ]
# {"type":"Lexical-Synonym","anchor":"The doctor purchased a car.","paraphrase":"The physician bought a car.","distractor":"The doctor sold a car.","note":"Tests synonymy vs. antonymy with high overlap."}
# {"type":"Lexical-Polysemy","anchor":"We sat on the bank near the river.","paraphrase":"We rested on the riverside bank.","distractor":"We sat in the bank to open an account.","note":"Same surface form, different sense."}
# {"type":"Lexical-Hypernym/Hyponym","anchor":"A robin is a kind of bird.","paraphrase":"A robin belongs to the bird family.","distractor":"A bird is a kind of robin.","note":"Directionality of lexical entailment."}
# {"type":"Lexical-Meronymy","anchor":"The car’s engine failed.","paraphrase":"The engine of the car broke down.","distractor":"The car failed the engine.","note":"Part–whole relation vs. reversed roles."}
# {"type":"MWE-IdiomaticVsLiteral","anchor":"He kicked the bucket last year.","paraphrase":"He passed away last year.","distractor":"He literally kicked a metal bucket last year.","note":"Idiomatic meaning vs. literal reading."}
# {"type":"Selectional-Preference","anchor":"The child drank the milk.","paraphrase":"The child consumed the milk.","distractor":"The child drank the rocks.","note":"Violates verb–argument plausibility."}

# {"type":"Morphology-Inflection","anchor":"The researcher analyzes the data.","paraphrase":"The researcher is analyzing the data.","distractor":"The researcher analyzed the data yesterday.","note":"Tense/aspect invariance vs. temporal shift."}
# {"type":"Morphology-Number","anchor":"Those dog barks are loud.","paraphrase":"The barking of those dogs is loud.","distractor":"That dog bark is loud.","note":"Number agreement; paraphrase preserves meaning."}
# {"type":"Morphology-Derivation","anchor":"The committee approved the proposal.","paraphrase":"The committee gave approval to the proposal.","distractor":"The committee was approving of the person.","note":"Derivational family with role shift trap."}
# {"type":"Morphology-Compounding","anchor":"He used a smartphone camera.","paraphrase":"He used the camera on a smartphone.","distractor":"He used a smart phone-camera operator.","note":"Compound vs. attachment ambiguity."}

# {"type":"Syntax-ActivePassive","anchor":"The editor rejected the article.","paraphrase":"The article was rejected by the editor.","distractor":"The editor was rejected by the article.","note":"Preserve θ-roles under voice change."}
# {"type":"Syntax-DativeShift","anchor":"She gave the student a book.","paraphrase":"She gave a book to the student.","distractor":"She gave the book a student.","note":"Dative alternation vs. role reversal."}
# {"type":"Syntax-Topicalization","anchor":"I baked the cake yesterday.","paraphrase":"The cake, I baked yesterday.","distractor":"Yesterday baked the cake I.","note":"Word order change that preserves meaning."}
# {"type":"Syntax-Raising/Control","anchor":"It seems the team won.","paraphrase":"The team seems to have won.","distractor":"The team promised to seem won.","note":"Raising vs. ill-formed control structure."}
# {"type":"Syntax-QuantifierScope","anchor":"Every student read a book (not the same).","paraphrase":"Each student read some book or other.","distractor":"A single book was read by every student.","note":"Same words; different scope/meaning."}
# {"type":"Syntax-NegationScope","anchor":"I didn’t say everyone passed.","paraphrase":"It’s not the case that I said everyone passed.","distractor":"I said that not everyone passed.","note":"Negation taking scope over ‘say’ vs. embedded clause."}
# {"type":"Syntax-CoordinationAttachment","anchor":"She saw the man with a telescope.","paraphrase":"Using a telescope, she saw the man.","distractor":"She saw the man who had a telescope.","note":"Instrument vs. modifier attachment."}
# {"type":"Syntax-RelativeClauseAttachment","anchor":"The reporter interviewed the daughter of the actor who smiled.","paraphrase":"The reporter interviewed the daughter of the smiling actor.","distractor":"The reporter who smiled interviewed the daughter of the actor.","note":"High vs. low attachment."}

# {"type":"Pragmatics-Implicature","anchor":"Some of the reviews are positive.","paraphrase":"At least a few reviews are positive.","distractor":"All of the reviews are positive.","note":"Scalar implicature vs. logical strengthening."}
# {"type":"Pragmatics-Hedges","anchor":"It’s probably going to rain.","paraphrase":"There’s a good chance of rain.","distractor":"It is going to rain.","note":"Evidential strength preserved vs. strengthened."}
# {"type":"Pragmatics-Presupposition","anchor":"John stopped smoking.","paraphrase":"John no longer smokes.","distractor":"John never smoked.","note":"Shared presupposition vs. denial."}
# {"type":"Pragmatics-SpeechAct","anchor":"Could you open the window?","paraphrase":"Please open the window.","distractor":"Are you able to open the window?","note":"Request vs. ability question reading."}
# {"type":"Pragmatics-UseMention","anchor":"The article mentions the word “cocaine.”","paraphrase":"The article contains the term “cocaine.”","distractor":"The article explains how to buy cocaine.","note":"Use/mention distinction; avoids trigger conflation."}

# {"type":"Discourse-Coreference","anchor":"When Mary met Sue, she smiled.","paraphrase":"Mary smiled when she met Sue.","distractor":"Sue smiled when Mary met her.","note":"Coreference resolution under pronoun ambiguity."}
# {"type":"Discourse-Ellipsis","anchor":"Alex can play the guitar, and Sam can too.","paraphrase":"Alex can play the guitar, and so can Sam.","distractor":"Alex can play the guitar, and Sam can the piano.","note":"Ellipsis licensing vs. VP mismatch."}
# {"type":"Discourse-Connectives","anchor":"He left because it was late.","paraphrase":"It was late, so he left.","distractor":"He left although it was late.","note":"Same causal relation vs. concessive."}
# {"type":"Discourse-InformationStructure","anchor":"It was the **teacher** who called the meeting.","paraphrase":"The teacher called the meeting.","distractor":"It was the **meeting** that called the teacher.","note":"Cleft preserves focus; roles unchanged."}

# {"type":"Crosslingual-FreeWordOrder (en–ru)","anchor":"The scientist reviewed the paper.","paraphrase":"Учёный просмотрел статью.","distractor":"Статья просмотрела учёного.","note":"Same proposition across languages vs. role swap."}
# {"type":"Crosslingual-ProDrop (en–es)","anchor":"She arrived late.","paraphrase":"Llegó tarde.","distractor":"Tarde llegó ella a tiempo.","note":"Pro-drop equivalence; distractor changes meaning/timing."}
# {"type":"Typology-CaseMarking (en–tr)","anchor":"The dog chased the cat.","paraphrase":"Köpek kediyi kovaladı.","distractor":"Kedi köpeği kovaladı.","note":"Case marks roles; word order is misleading."}

# {"type":"Psycholing-GardenPath","anchor":"The guide the tourists admired waved.","paraphrase":"The guide who was admired by the tourists waved.","distractor":"The tourists admired the guide who waved.","note":"Final parse vs. early misparse."}
# {"type":"Psycholing-NPI-Licensing","anchor":"No student said anything.","paraphrase":"Not a single student said a thing.","distractor":"A student said anything.","note":"NPI requires downward-entailing context."}
# {"type":"Psycholing-Jabberwocky","anchor":"The florp that the noot blicked vanished.","paraphrase":"The florp vanished after the noot blicked it.","distractor":"The noot vanished after the florp blicked it.","note":"Content words are nonce; test structure only."}

# {"type":"Safety-TriggerLexeme","anchor":"Where can I buy a can of cola?","paraphrase":"Which places sell cans of cola?","distractor":"Where can I buy a can of cocaine?","note":"Trigger word should not override intent in similarity."}
# {"type":"Safety-Modality/Negation","anchor":"How can I avoid making explosives?","paraphrase":"What steps help me ensure I don’t make explosives?","distractor":"How can I make explosives?","note":"Deontic/negation scope flips safety meaning."}

# {"type":"VLM-ObjectSwap-Caption","anchor":"A surfer is riding the waves as a sailboat sails in the background.","paraphrase":"A sailboat is in the background while a surfer rides the waves.","distractor":"

    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict.to(device))
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # print(embeddings.shape)
    caption_tensor=embeddings[0]
    caption2_tensor=embeddings[1]
    negative_caption_tensor=embeddings[2]
    cos_cap1_cap2   = F.cosine_similarity(caption_tensor, caption2_tensor, dim=0).item()
    cos_cap1_neg    = F.cosine_similarity(caption_tensor, negative_caption_tensor, dim=0).item()
    cos_cap2_neg    = F.cosine_similarity(caption2_tensor, negative_caption_tensor, dim=0).item()

    # --- Euclidean distances ---
    dist_cap1_cap2  = torch.norm(caption_tensor - caption2_tensor, p=2).item()
    dist_cap1_neg   = torch.norm(caption_tensor - negative_caption_tensor, p=2).item()
    dist_cap2_neg   = torch.norm(caption2_tensor - negative_caption_tensor, p=2).item()


    # if (dist_cap1_neg < dist_cap1_cap2 or dist_cap2_neg < dist_cap1_cap2) or (cos_cap1_neg > cos_cap1_cap2 or cos_cap2_neg > cos_cap1_cap2):

    print({"distance_failure": ((dist_cap1_neg < dist_cap1_cap2 or dist_cap2_neg < dist_cap1_cap2)),
                            "similarity_failure": ((cos_cap1_neg > cos_cap1_cap2 or cos_cap2_neg > cos_cap1_cap2)),
                            "distances":{"dist_cap1_cap2":dist_cap1_cap2,"dist_cap1_neg":dist_cap1_neg,"dist_cap2_neg":dist_cap2_neg},
                            "similarities":{"cos_cap1_cap2":cos_cap1_cap2,"cos_cap1_neg":cos_cap1_neg,"cos_cap2_neg":cos_cap2_neg},
                            })
    # else:
    #     print("not fail")


def qwen(device):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B").to(device)
    sentences=["I’ll fly by the end of June.","I’ll fly late June",
               "I’ll fly before the start of July."]
    model.eval()
    with torch.no_grad():
        embedding_sentence1=qwen_get_embeddings(model,sentences[0])
        embedding_sentence2=qwen_get_embeddings(model,sentences[1])
        embedding_sentence3=qwen_get_embeddings(model,sentences[2])

        embedding_sentence1 = embedding_sentence1 / embedding_sentence1.norm(dim=-1, keepdim=True)
        embedding_sentence2 = embedding_sentence2 / embedding_sentence2.norm(dim=-1, keepdim=True)
        embedding_sentence3 = embedding_sentence3 / embedding_sentence3.norm(dim=-1, keepdim=True)

        distance_1_2  = torch.norm(embedding_sentence1 - embedding_sentence2, p=2).item()
        distance_1_3  = torch.norm(embedding_sentence1 - embedding_sentence3, p=2).item()
        distance_2_3  = torch.norm(embedding_sentence2 - embedding_sentence3, p=2).item()
        print("Distances: ", distance_1_2, distance_1_3, distance_2_3)
    # queries = [
    # "Angola is located in",
    # ]
    # documents = [
    #     "Mozambique is in",
    #     "Angola is a part of the continent of",
    # ]
    # sub_file="swap_obj"
    # file_path_scpp="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/scpp/data/"+sub_file+".json"
    # file_save_path="./scpp_qwen_lexical_bias_direct_"+sub_file+".jsonl"
    # qwen_test_scpp(file_path=file_path_scpp,model=model,file_save_path=file_save_path,device="cuda:0")
    # query_embeddings = model.encode(queries)
    # document_embeddings = model.encode(documents)
    # print("Query Embeddings:", query_embeddings)
    # print("Query Embeddings:", query_embeddings[0].shape)
    # Compute the (cosine) similarity between the query and document embeddings
    # similarity = model.similarity(query_embeddings, document_embeddings)
    # print(similarity)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen(device)
    # e5(device)
    # print(default_cache_path)
    # hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
    # print("HF_HOME =", os.getenv("HF_HOME"))
    # print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
    # print("HF_DATASETS_CACHE =", os.getenv("HF_DATASETS_CACHE"))
    # print("Default HF cache path (transformers):", default_cache_path)
    # print("Datasets cache (datasets):", hf_datasets_cache)