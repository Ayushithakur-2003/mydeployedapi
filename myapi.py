from fastapi import FastAPI
from transformers import pipeline
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

model_id = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model_id)
cl = classifier
vec_com = ["technical issues", "payment problem", "return", "difficulties", "refund"]

vec_dev_1 = ["quality","price", "features", "shipping", "reliability", "durability", "performance", "user experience"]

vec_sk_1 = ["skin care","quality", "skin", "skin type", "effects", "texture", "fragrance"]

vec_cl_1 = ["material","quality", "color", "clothes", "size"]

@app.get("/{domain}/{feedback}")
async def root(domain, feedback):
 if domain == "device":
       candidate_labels_product = vec_dev_1
 elif domain == "skin care products":
       candidate_labels_product = vec_sk_1
 elif domain == "clothing":
       candidate_labels_product = vec_cl_1
 candidate_labels_experience = ["ease of use", "satisfaction", "dissatisfaction", "customer service", "inconvenience"]
 candidate_labels_issue = ["technical issues", "payment problem", "return", "difficulties", "refund"]

 product_aspects = classifier(feedback, candidate_labels=candidate_labels_product, multilabel=True)
 customer_experience = classifier(feedback, candidate_labels=candidate_labels_experience, multilabel=True)
 issue = classifier(feedback, candidate_labels=candidate_labels_issue, multilabel=True)
         
 aspect_res = ("aspect :", product_aspects["labels"][0], product_aspects["labels"][1])
 s1 =  (round(product_aspects["scores"][0]*100, 2) + round(product_aspects["scores"][1]*100, 2))/2
 aspect_sc = ("score :", s1, "%")   

 ce_res =("customer experience :", customer_experience["labels"][0], customer_experience["labels"][1])
 s2 =  (round(customer_experience["scores"][0]*100, 2) + round(customer_experience["scores"][1]*100, 2))/2
 ce_sc =("score :", s2, "%") 

 s = feedback.split()
 if "technical issues" in s or "payment problem" in s or "return" in s or "difficulties" in s or "refund" in s:
       iss_res =("refund/return/transtaction issues :", issue["labels"][0], issue["labels"][1])
       s3 =  (round(issue["scores"][0]*100, 2) + round(issue["scores"][1]*100, 2))/2
       iss_sc =("score :", s3, "%")
 else:
       iss_res =("no refund/return/transtaction issues")

 response_data = {'Feedback': feedback,
                  'analysis': [aspect_res, ce_res, iss_res]
                        } 
 return response_data 