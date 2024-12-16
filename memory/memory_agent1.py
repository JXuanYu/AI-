from agentdriver.llm_core.timeout import timeout
from transformers import (BertModel, BertTokenizer, AutoTokenizer, 
                          DPRContextEncoder, AutoModel, RealmEmbedder, RealmForOpenQA,
                          RagTokenizer, RagRetriever, RagSequenceForGeneration)
from agentdriver.memory.common_sense_memory import CommonSenseMemory
from agentdriver.memory.experience_memory import ExperienceMemory
import torch
from torch import nn

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Using the pooled output of BERT
        return pooled_output

class ClassificationNetwork(nn.Module):
    def __init__(self, num_labels):
        super(ClassificationNetwork, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

class DPRNetwork(nn.Module):
    def __init__(self):
        super(DPRNetwork, self).__init__()
        self.bert = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to("cuda")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

class ANCENetwork(nn.Module):
    def __init__(self):
        super(ANCENetwork, self).__init__()
        self.bert = AutoModel.from_pretrained("castorini/ance-dpr-question-multi").to("cuda")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output
    
class REALMNetwork(nn.Module):
    def __init__(self):
        super(REALMNetwork, self).__init__()
        self.bert = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder").realm.to("cuda")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

class BGENetwork(nn.Module):
    def __init__(self):
        super(BGENetwork, self).__init__()
        self.bert = AutoModel.from_pretrained("BAAI/bge-large-en").to("cuda")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

class ORQANetwork(nn.Module):
    def __init__(self):
        super(ORQANetwork, self).__init__()
        self.bert = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa").embedder.realm.to("cuda")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

class BM25Network(nn.Module):
    def __init__(self):
        super(BM25Network, self).__init__()
        self.bert = AutoModel.from_pretrained("facebook/spar-wiki-bm25-lexmodel-context-encoder").to("cuda")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

class CommonSenseMemory:
    def __init__(self) -> None:
        self.common_sense = {
            "Traffic Rules": TRAFFIC_RULES,
        }

        # Initialize RAG components
        self.rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", passages_path="path_to_passages")
        self.rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=self.rag_retriever)

    def retrieve(self, knowledge_types: list = None):
        commonsense_prompt = "\n"
        if knowledge_types is not None:
            for knowledge_type in knowledge_types:
                commonsense_prompt += ("*"*5 + knowledge_type + ":" + "*"*5 + "\n")
                for rule in self.common_sense[knowledge_type]:
                    commonsense_prompt += ("- " + rule + "\n")
        else: # fetch all knowledge
            for knowledge_type in self.common_sense.keys():
                commonsense_prompt += ("*"*5 + knowledge_type + ":" + "*"*5 + "\n")
                for rule in self.common_sense[knowledge_type]:
                    commonsense_prompt += ("- " + rule + "\n")
        return commonsense_prompt

    def retrieve_with_rag(self, query: str):
        inputs = self.rag_tokenizer(query, return_tensors="pt").to("cuda")
        outputs = self.rag_model.generate(**inputs)
        return self.rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)

TRAFFIC_RULES = [
    "Avoid collision with other objects.",
    "Always drive on drivable regions.",
    "Avoid driving on occupied regions.",
    "Pay attention to your ego-states and historical trajectory when planning.",
    "Maintain a safe distance from the objects in front of you.",
    "When passing a gas station, be sure to slow down, avoid rapid acceleration or overtaking.",
    "Try to avoid following novice drivers, and maintain a safe distance.",
    "On speed-limited sections, you must follow the speed limit regulations and keep your speed within a reasonable range.",
    "When driving at night, avoid driving on white roads as much as possible, and if you must, be extremely cautious.",
    "When driving in the rain, reduce your speed and maintain a safe distance.",
    "When turning, do not exceed a speed of 30 km/h.",
    "When following a car, maintain a distance of at least one car length and frequently check your rearview mirror.",
    "When driving on special road sections such as slopes or bridges, maintain sufficient distance to avoid the need for emergency evasive actions.",
    "At night, try to drive in the middle lane to facilitate turning or avoiding obstacles when necessary.",
    "Upon entering a residential area, observe the surroundings carefully and avoid sudden acceleration or abrupt turns if there are children, elderly people, or obstacles ahead.",
    "When encountering buses, taxis, electric vehicles, or large trucks, maintain a safe distance to avoid close parallel driving.",
    "Do not change lanes frequently while driving; maintain a consistent speed.",
    "When approaching intersections or obstructions, slow down and be prepared to brake at any time.",
    "Do not frequently change lanes or make sharp turns at intersections; driving straight is the safest method.",
    "Yielding to pedestrians is a basic responsibility of every driver.",
    "Do not overtake at red light intersections to avoid potential danger.",
    "For vehicles traveling in the same direction, the left vehicle yields to the right; safety is the priority.",
    "In urban areas, avoid using high beams, turn off high beams when meeting other vehicles, and use high beams to alert the vehicle ahead in no-horn zones.",
    "Avoid driving alongside large trucks for extended periods; observe the truck's movements before overtaking to ensure safety.",
    "Do not overtake at intersections, especially avoid the large truck's right-side blind spot.",
    "Braking over speed bumps can damage the suspension system; slow down in advance and release the brake before crossing the bump.",
    "Always use the turn signal when changing lanes to inform other drivers of your intentions.",
    "Right turns must yield to left turns, and slow down when entering or exiting tunnels or making any turns.",
    "On highways, do not drive in the overtaking lane (leftmost lane) for extended periods.",
    "Turn on the turn signal in advance when turning or changing lanes.",
    "Adjust your headlights in advance when meeting other vehicles.",
    "Always slow down at intersections.",
    "Do not overtake recklessly when the large bus ahead slows down or stops.",
    "When the brake lights of the vehicle ahead are on, follow with light braking.",
    "Change to the appropriate lane in advance before making a turn.",
    "Do not drive alongside large vehicles for long periods.",
    "Maintaining a safe distance is crucial.",
]


class MemoryAgent:
    def __init__(self, data_path, model_name="gpt-3.5-turbo-0613", verbose=False, compare_perception=False, embedding="Linear", args=None) -> None:
        self.model_name = model_name
        self.common_sense_memory = CommonSenseMemory()
        self.embedding = embedding
        
        # Initialize RAG components
        self.rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", passages_path="path_to_passages")
        self.rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=self.rag_retriever)

        if self.embedding == "Contrastive":
            embedder_dir = 'RAG/embedder/contrastive_embedder_user_random_diverse/checkpoint-300'
            self.embedding_model = TripletNetwork().to("cuda")
            self.embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.embedding_model.load_state_dict(torch.load(embedder_dir + "/pytorch_model.bin"))
            self.embedding_model.eval()
        elif self.embedding == "Classification":
            embedder_dir = 'RAG/embedder/classification_embedder_user/checkpoint-500'
            num_labels = 11
            self.embedding_model = ClassificationNetwork(num_labels=num_labels).to("cuda")
            self.embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.embedding_model.load_state_dict(torch.load(embedder_dir + "/pytorch_model.bin"))
            self.embedding_model.eval()
        elif self.embedding == "facebook/dpr-ctx_encoder-single-nq-base":
            self.embedding_model = DPRNetwork().to("cuda")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            self.embedding_model.eval()
        elif self.embedding == "castorini/ance-dpr-question-multi":
            self.embedding_model = ANCENetwork().to("cuda")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("castorini/ance-dpr-question-multi")
            self.embedding_model.eval()
        elif self.embedding == "bge-large-en":
            self.embedding_model = BGENetwork().to("cuda")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en")
            self.embedding_model.eval()
        elif self.embedding == "realm-cc-news-pretrained-embedder":
            self.embedding_model = REALMNetwork().to("cuda")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-embedder")
            self.embedding_model.eval()
        elif self.embedding == "realm-orqa-nq-openqa":
            self.embedding_model = ORQANetwork().to("cuda")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
            self.embedding_model.eval()
        elif self.embedding == "spar-wiki-bm25-lexmodel-context-encoder":
            self.embedding_model = BM25Network().to("cuda")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("facebook/spar-wiki-bm25-lexmodel-context-encoder")
            self.embedding_model.eval()
        else:
            self.embedding_model = None
            self.embedding_tokenizer = None

        self.experience_memory = ExperienceMemory(data_path, model_name=self.model_name, verbose=verbose, compare_perception=compare_perception, embedding=self.embedding, embedding_model=self.embedding_model, embedding_tokenizer=self.embedding_tokenizer, args=args)
        self.verbose = verbose

    def retrieve(self, working_memory):
        raise NotImplementedError
    
    def retrieve_common_sense_memory(self, knowledge_types: list = None):
        return self.common_sense_memory.retrieve(knowledge_types=knowledge_types)
    
    def retrieve_common_sense_with_rag(self, query: str):
        return self.common_sense_memory.retrieve_with_rag(query)

    def retrieve_experience_memory(self, working_memory, embedding):
        return self.experience_memory.retrieve(working_memory)

    def insert(self, working_memory):
        raise NotImplementedError

    def update(self, working_memory):
        raise NotImplementedError

    @timeout(15)
    def run(self, working_memory):
        common_sense_prompts = self.retrieve_common_sense_with_rag("Provide common sense information related to the context")  # Example query
        experience_prompt = self.retrieve_experience_memory(working_memory, self.embedding)

        return common_sense_prompts, experience_prompt
