---
language:
- en
pipeline_tag: text-generation
base_model: meta-llama/Meta-Llama-3.1-8B
tags:
- facebook
- meta
- pytorch
- llama
- llama-3
license: llama3.1
extra_gated_prompt: >-
  ### LLAMA 3.1 COMMUNITY LICENSE AGREEMENT

  Llama 3.1 Version Release Date: July 23, 2024
  
  "Agreement" means the terms and conditions for use, reproduction, distribution and modification of the 
  Llama Materials set forth herein.

  "Documentation" means the specifications, manuals and documentation accompanying Llama 3.1
  distributed by Meta at https://llama.meta.com/doc/overview.

  "Licensee" or "you" means you, or your employer or any other person or entity (if you are entering into
  this Agreement on such person or entity’s behalf), of the age required under applicable laws, rules or
  regulations to provide legal consent and that has legal authority to bind your employer or such other
  person or entity if you are entering in this Agreement on their behalf.

  "Llama 3.1" means the foundational large language models and software and algorithms, including
  machine-learning model code, trained model weights, inference-enabling code, training-enabling code,
  fine-tuning enabling code and other elements of the foregoing distributed by Meta at
  https://llama.meta.com/llama-downloads.

  "Llama Materials" means, collectively, Meta’s proprietary Llama 3.1 and Documentation (and any
  portion thereof) made available under this Agreement.

  "Meta" or "we" means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your
  principal place of business is in the EEA or Switzerland) and Meta Platforms, Inc. (if you are located
  outside of the EEA or Switzerland).
     
  1. License Rights and Redistribution.

  a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free
  limited license under Meta’s intellectual property or other rights owned by Meta embodied in the Llama
  Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the
  Llama Materials.

  b. Redistribution and Use.

  i. If you distribute or make available the Llama Materials (or any derivative works
  thereof), or a product or service (including another AI model) that contains any of them, you shall (A)
  provide a copy of this Agreement with any such Llama Materials; and (B) prominently display “Built with
  Llama” on a related website, user interface, blogpost, about page, or product documentation. If you use
  the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or
  otherwise improve an AI model, which is distributed or made available, you shall also include “Llama” at
  the beginning of any such AI model name.

  ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as part 
  of an integrated end user product, then Section 2 of this Agreement will not apply to you.

  iii. You must retain in all copies of the Llama Materials that you distribute the following
  attribution notice within a “Notice” text file distributed as a part of such copies: “Llama 3.1 is
  licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights
  Reserved.”

  iv. Your use of the Llama Materials must comply with applicable laws and regulations
  (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Llama
  Materials (available at https://llama.meta.com/llama3_1/use-policy), which is hereby incorporated by
  reference into this Agreement.

  2. Additional Commercial Terms. If, on the Llama 3.1 version release date, the monthly active users
  of the products or services made available by or for Licensee, or Licensee’s affiliates, is greater than 700
  million monthly active users in the preceding calendar month, you must request a license from Meta,
  which Meta may grant to you in its sole discretion, and you are not authorized to exercise any of the
  rights under this Agreement unless or until Meta otherwise expressly grants you such rights.

  3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY
  OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF
  ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED,
  INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT,
  MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR
  DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE LLAMA MATERIALS AND
  ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND
  RESULTS.

  4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING
  OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL,
  INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED
  OF THE POSSIBILITY OF ANY OF THE FOREGOING.

  5. Intellectual Property.

  a. No trademark licenses are granted under this Agreement, and in connection with the Llama
  Materials, neither Meta nor Licensee may use any name or mark owned by or associated with the other
  or any of its affiliates, except as required for reasonable and customary use in describing and
  redistributing the Llama Materials or as set forth in this Section 5(a). Meta hereby grants you a license to
  use “Llama” (the “Mark”) solely as required to comply with the last sentence of Section 1.b.i. You will
  comply with Meta’s brand guidelines (currently accessible at
  https://about.meta.com/brand/resources/meta/company-brand/ ). All goodwill arising out of your use
  of the Mark will inure to the benefit of Meta.

  b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with
  respect to any derivative works and modifications of the Llama Materials that are made by you, as
  between you and Meta, you are and will be the owner of such derivative works and modifications.

  c. If you institute litigation or other proceedings against Meta or any entity (including a
  cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Llama 3.1 outputs or
  results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other
  rights owned or licensable by you, then any licenses granted to you under this Agreement shall
  terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold
  harmless Meta from and against any claim by any third party arising out of or related to your use or
  distribution of the Llama Materials.

  6. Term and Termination. The term of this Agreement will commence upon your acceptance of this
  Agreement or access to the Llama Materials and will continue in full force and effect until terminated in
  accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in
  breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete
  and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the termination of this
  Agreement.

  7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of
  the State of California without regard to choice of law principles, and the UN Convention on Contracts
  for the International Sale of Goods does not apply to this Agreement. The courts of California shall have
  exclusive jurisdiction of any dispute arising out of this Agreement.

  ### Llama 3.1 Acceptable Use Policy

  Meta is committed to promoting safe and fair use of its tools and features, including Llama 3.1. If you
  access or use Llama 3.1, you agree to this Acceptable Use Policy (“Policy”). The most recent copy of
  this policy can be found at [https://llama.meta.com/llama3_1/use-policy](https://llama.meta.com/llama3_1/use-policy)

  #### Prohibited Uses

  We want everyone to use Llama 3.1 safely and responsibly. You agree you will not use, or allow
  others to use, Llama 3.1 to:
   1. Violate the law or others’ rights, including to:
      1. Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content, such as:
          1. Violence or terrorism
          2. Exploitation or harm to children, including the solicitation, creation, acquisition, or dissemination of child exploitative content or failure to report Child Sexual Abuse Material
          3. Human trafficking, exploitation, and sexual violence
          4. The illegal distribution of information or materials to minors, including obscene materials, or failure to employ legally required age-gating in connection with such information or materials.
          5. Sexual solicitation
          6. Any other criminal activity
      3. Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals
      4. Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other essential goods and services
      5. Engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices
      6. Collect, process, disclose, generate, or infer health, demographic, or other sensitive personal or private information about individuals without rights and consents required by applicable laws
      7. Engage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any products or services using the Llama Materials
      8. Create, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity, operation or appearance of a website or computer system
  2. Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including use of Llama 3.1 related to the following:
      1. Military, warfare, nuclear industries or applications, espionage, use for materials or activities that are subject to the International Traffic Arms Regulations (ITAR) maintained by the United States Department of State
      2. Guns and illegal weapons (including weapon development)
      3. Illegal drugs and regulated/controlled substances
      4. Operation of critical infrastructure, transportation technologies, or heavy machinery
      5. Self-harm or harm to others, including suicide, cutting, and eating disorders
      6. Any content intended to incite or promote violence, abuse, or any infliction of bodily harm to an individual
  3. Intentionally deceive or mislead others, including use of Llama 3.1 related to the following:
      1. Generating, promoting, or furthering fraud or the creation or promotion of disinformation
      2. Generating, promoting, or furthering defamatory content, including the creation of defamatory statements, images, or other content
      3. Generating, promoting, or further distributing spam
      4. Impersonating another individual without consent, authorization, or legal right
      5. Representing that the use of Llama 3.1 or outputs are human-generated
      6. Generating or facilitating false online engagement, including fake reviews and other means of fake online engagement
  4. Fail to appropriately disclose to end users any known dangers of your AI system
  
  Please report any violation of this Policy, software “bug,” or other problems that could lead to a violation
  of this Policy through one of the following means:
      * Reporting issues with the model: [https://github.com/meta-llama/llama-models/issues](https://github.com/meta-llama/llama-models/issues)
      * Reporting risky content generated by the model:
      developers.facebook.com/llama_output_feedback
      * Reporting bugs and security concerns: facebook.com/whitehat/info
      * Reporting violations of the Acceptable Use Policy or unlicensed uses of Meta Llama 3: LlamaUseReport@meta.com
extra_gated_fields:
  First Name: text
  Last Name: text
  Date of birth: date_picker
  Country: country
  Affiliation: text
  Job title:
    type: select
    options: 
      - Student
      - Research Graduate
      - AI researcher
      - AI developer/engineer
      - Reporter
      - Other  
  geo: ip_location  
  By clicking Submit below I accept the terms of the license and acknowledge that the information I provide will be collected stored processed and shared in accordance with the Meta Privacy Policy: checkbox
extra_gated_description: The information you provide will be collected, stored, processed and shared in accordance with the [Meta Privacy Policy](https://www.facebook.com/privacy/policy/).
extra_gated_button_content: Submit
---


# Model Details

Llama Guard 3 is a Llama-3.1-8B pretrained model, fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM – it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.

Llama Guard 3 was aligned to safeguard against the MLCommons standardized hazards taxonomy and designed to support Llama 3.1 capabilities. Specifically, it provides content moderation in 8 languages, and was optimized to support safety and security for search and code interpreter tool calls.

Below is a response classification example for Llama Guard 3.


<p align="center">
  <img src="llama_guard_3_figure.png" width="800"/>
</p>

In order to produce classifier scores, we look at the probability for the first token, and use that as the “unsafe” class probability. We can then apply score thresholding to make binary decisions.

# Hazard Taxonomy and Policy

The model is trained to predict safety labels on the 14 categories shown below, based on the [MLCommons taxonomy](https://mlcommons.org/2024/04/mlc-aisafety-v0-5-poc/) of 13 hazards, as well as an additional category for Code Interpreter Abuse for tool calls use cases

<table align="center">
<thead>
  <tr>
    <th colspan="2">Hazard categories</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>S1: Violent Crimes</td>
    <td>S2: Non-Violent Crimes</td>
  </tr>
  <tr>
    <td>S3: Sex-Related Crimes</td>
    <td>S4: Child Sexual Exploitation</td>
  </tr>
  <tr>
    <td>S5: Defamation</td>
    <td>S6: Specialized Advice</td>
  </tr>
  <tr>
    <td>S7: Privacy</td>
    <td>S8: Intellectual Property</td>
  </tr>
  <tr>
    <td>S9: Indiscriminate Weapons</td>
    <td>S10: Hate</td>
  </tr>
  <tr>
    <td>S11: Suicide &amp; Self-Harm</td>
    <td>S12: Sexual Content</td>
  </tr>
  <tr>
    <td>S13: Elections</td>
    <td>S14: Code Interpreter Abuse</td>
  </tr>
</tbody>
</table>

**S1 - Violent Crimes**

Responses that enable, encourage, or endorse the commission of violent crimes, including: (1) unlawful violence toward people (ex: terrorism, genocide, murder, hate-motivated violent crimes, child abuse, assault, battery, kidnapping) and (2) unlawful violence toward animals (ex: animal abuse)

**S2 - Non-Violent Crimes**

Responses that enable, encourage, or endorse the commission of non-violent crimes, including: (1) personal crimes (ex: labor trafficking, threats, intimidation, hate-motivated non-violent crimes), (2) financial crimes (ex: fraud, scams, money laundering), (3) property crimes (ex: theft, arson, vandalism), (4) drug crimes (ex: creating or using regulated substances), (5) weapons crimes (ex: producing unlicensed firearms), and (6) cyber crimes (ex: hacking)

**S3 - Sex-Related Crimes**

Responses that enable, encourage, or endorse the commission of sex-related crimes, including: (1) sex trafficking, (2) sexual assault (ex: rape), (3) sexual harassment (physical, verbal, or visual), and (4) prostitution

**S4 - Child Sexual Exploitation**

Responses that contain, describe, enable, encourage, or endorse the sexual abuse of children

**S5 - Defamation**

Responses that are both verifiably false and likely to injure a living person’s reputation

**S6 - Specialized Advice**

Responses that contain specialized financial, medical, or legal advice, or that indicate dangerous activities or objects are safe

**S7 - Privacy**

Responses that contain sensitive, nonpublic personal information that could undermine someone’s physical, digital, or financial security

**S8 - Intellectual Property**

Responses that may violate the intellectual property rights of any third party

**S9 - Indiscriminate Weapons**

Responses that enable, encourage, or endorse the creation of indiscriminate weapons, including: (1) chemical weapons (ex: nerve gas), (2) biological weapons (ex: anthrax), (3) radiological weapons (ex: cobalt bombs), (4) nuclear weapons (ex: fission bombs), and (5) high-yield explosive weapons (ex: cluster munitions)

**S10 - Hate**

Responses that demean or dehumanize people on the basis of their sensitive, personal characteristics (i.e., race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and/or serious disease)

**S11 - Suicide &amp; Self-Harm**

Responses that enable, encourage, or endorse acts of intentional self-harm, including: (1) suicide, (2) self-injury (ex: cutting), and (3) disordered eating

**S12 - Sexual Content**

Responses that contain erotica

**S13 - Elections**

Responses that contain factually incorrect information about electoral systems and processes, including in the time, place, or manner of voting in civic elections

**S14 - Code Interpreter Abuse**

Responses that seek to abuse code interpreters, including those that enable denial of service attacks, container escapes or privilege escalation exploits


# Supported languages

Llama Guard 3 supports content safety for the following languages : English, French, German, Hindi, Italian, Portuguese, Spanish, Thai.

# Usage

> [!IMPORTANT]
>
> This repository corresponds to half-precision version of the model. A 8-bit precision version is also provided, please visit [meta-llama/Llama-Guard-3-8B-INT8](https://huggingface.co/meta-llama/Llama-Guard-3-8B-INT8).

Llama Guard 3 can be directly used with `transformers`. It is only supported since `transformers` version 4.43.

  ```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-Guard-3-8B"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
  input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
  output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
  prompt_len = input_ids.shape[-1]
  return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

moderate([
  {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
  {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
])
```

# Training Data

We use the English data used by Llama Guard [1], which are obtained by getting Llama 2 and Llama 3 generations on prompts from the hh-rlhf dataset [2]. In order to scale training data for new categories and new capabilities such as multilingual and tool use, we collect additional human and synthetically generated data. Similar to the English data, the multilingual data are Human-AI conversation data that are either single-turn or multi-turn. To reduce the model’s false positive rate, we curate a set of multilingual benign prompt and response data where LLMs likely reject the prompts.

For the tool use capability, we consider search tool calls and code interpreter abuse. To develop training data for search tool use, we use Llama3 to generate responses to a collected and synthetic set of prompts. The generations are based on the query results obtained from the Brave Search API. To develop synthetic training data to detect code interpreter attacks, we use an LLM to generate safe and unsafe prompts.  Then, we use a non-safety-tuned LLM to generate code interpreter completions that comply with these instructions.  For safe data, we focus on data close to the boundary of what would be considered unsafe, to minimize false positives on such borderline examples.

# Evaluation

**Note on evaluations:** As discussed in the original Llama Guard paper, comparing model performance is not straightforward as each model is built on its own policy and is expected to perform better on an evaluation dataset with a policy aligned to the model. This highlights the need for industry standards. By aligning the Llama Guard family of models with the Proof of Concept MLCommons taxonomy of hazards, we hope to drive adoption of industry standards like this and facilitate collaboration and transparency in the LLM safety and content evaluation space.

In this regard, we evaluate the performance of Llama Guard 3 on MLCommons hazard taxonomy and compare it across languages with Llama Guard 2 [3] on our internal test. We also add GPT4 as baseline with zero-shot prompting using MLCommons hazard taxonomy.

Tables 1, 2, and 3 show that Llama Guard 3 improves over Llama Guard 2 and outperforms GPT4 in English, multilingual, and tool use capabilities. Noteworthily,  Llama Guard 3 achieves better performance with much lower false positive rates. We also benchmark Llama Guard 3 in the OSS dataset XSTest [4] and observe that it achieves the same F1 score but a lower false positive rate compared to Llama Guard 2.

<div align="center">
<small> Table 1: Comparison of performance of various models measured on our internal English test set for MLCommons hazard taxonomy (response classification).</small>

|                | **F1 ↑** | **AUPRC ↑** | **False Positive<br>Rate ↓** |
|--------------------------|:--------:|:-----------:|:----------------------------:|
| Llama Guard 2            |  0.877 |   0.927   |          0.081          |
| Llama Guard 3            |  0.939 |   0.985   |          0.040          |
| GPT4                     |  0.805 |    N/A    |          0.152          |
</div>

<br>

<table align="center">
<small><center>Table 2: Comparison of multilingual performance of various models measured on our internal test set for MLCommons hazard taxonomy (prompt+response classification).</center></small>
<thead>
  <tr>
    <th colspan="8"><center>F1 ↑ / FPR ↓</center></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td><center>French</center></td>
    <td><center>German</center></td>
    <td><center>Hindi</center></td>
    <td><center>Italian</center></td>
    <td><center>Portuguese</center></td>
    <td><center>Spanish</center></td>
    <td><center>Thai</center></td>
  </tr>
  <tr>
    <td>Llama Guard 2</td>
    <td><center>0.911/0.012</center></td>
    <td><center>0.795/0.062</center></td>
    <td><center>0.832/0.062</center></td>
    <td><center>0.681/0.039</center></td>
    <td><center>0.845/0.032</center></td>
    <td><center>0.876/0.001</center></td>
    <td><center>0.822/0.078</center></td>
  </tr>
  <tr>
    <td>Llama Guard 3</td>
    <td><center>0.943/0.036</center></td>
    <td><center>0.877/0.032</center></td>
    <td><center>0.871/0.050</center></td>
    <td><center>0.873/0.038</center></td>
    <td><center>0.860/0.060</center></td>
    <td><center>0.875/0.023</center></td>
    <td><center>0.834/0.030</center></td>
  </tr>
  <tr>
    <td>GPT4</td>
    <td><center>0.795/0.157</center></td>
    <td><center>0.691/0.123</center></td>
    <td><center>0.709/0.206</center></td>
    <td><center>0.753/0.204</center></td>
    <td><center>0.738/0.207</center></td>
    <td><center>0.711/0.169</center></td>
    <td><center>0.688/0.168</center></td>
  </tr>
</tbody>
</table>

<br>

<table align="center">
<small><center>Table 3: Comparison of performance of various models measured on our internal test set for other moderation capabilities (prompt+response classification).</center></small>
<thead>
  <tr>
    <th></th>
    <th colspan="3">Search tool calls</th>
     <th colspan="3">Code interpreter abuse</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td><center>F1 ↑</center></td>
    <td><center>AUPRC ↑</center></td>
    <td><center>FPR ↓</center></td>
    <td><center>F1 ↑</center></td>
    <td><center>AUPRC ↑</center></td>
    <td><center>FPR ↓</center></td>
  </tr>
  <tr>
    <td>Llama Guard 2</td>
    <td><center>0.749</center></td>
    <td><center>0.794</center></td>
    <td><center>0.284</center></td>
    <td><center>0.683</center></td>
    <td><center>0.677</center></td>
    <td><center>0.670</center></td>
  </tr>
  <tr>
    <td>Llama Guard 3</td>
    <td><center>0.856</center></td>
    <td><center>0.938</center></td>
    <td><center>0.174</center></td>
    <td><center>0.885</center></td>
    <td><center>0.967</center></td>
    <td><center>0.125</center></td>
  </tr>
  <tr>
    <td>GPT4</td>
    <td><center>0.732</center></td>
    <td><center>N/A</center></td>
    <td><center>0.525</center></td>
    <td><center>0.636</center></td>
    <td><center>N/A</center></td>
    <td><center>0.90</center></td>
  </tr>
</tbody>
</table>

# Application

As outlined in the Llama 3 paper, Llama Guard 3 provides industry leading system-level safety performance and is recommended to be deployed along with Llama 3.1. Note that, while deploying Llama Guard 3 will likely improve the safety of your system, it might increase refusals to benign prompts (False Positives). Violation rate improvement and impact on false positives as measured on internal benchmarks are provided in the Llama 3 paper.

# Quantization

We are committed to help the community deploy Llama systems responsibly. We provide a quantized version of Llama Guard 3 to lower the deployment cost. We used int 8 [implementation](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) integrated into the hugging face ecosystem, reducing the checkpoint size by about 40% with very small impact on model performance. In Table 5, we observe that the performance quantized model is comparable to the original model.

<table align="center">
<small><center>Table 5: Impact of quantization on Llama Guard 3 performance.</center></small>
<tbody>
<tr>
<td rowspan="2"><br />
<p><span>Task</span></p>
</td>
<td rowspan="2"><br />
<p><span>Capability</span></p>
</td>
<td colspan="4">
<p><center><span>Non-Quantized</span></center></p>
</td>
<td colspan="4">
<p><center><span>Quantized</span></center></p>
</td>
</tr>
<tr>
<td>
<p><span>Precision</span></p>
</td>
<td>
<p><span>Recall</span></p>
</td>
<td>
<p><span>F1</span></p>
</td>
<td>
<p><span>FPR</span></p>
</td>
<td>
<p><span>Precision</span></p>
</td>
<td>
<p><span>Recall</span></p>
</td>
<td>
<p><span>F1</span></p>
</td>
<td>
<p><span>FPR</span></p>
</td>
</tr>
<tr>
<td rowspan="3">
<p><span>Prompt Classification</span></p>
</td>
<td>
<p><span>English</span></p>
</td>
<td>
<p><span>0.952</span></p>
</td>
<td>
<p><span>0.943</span></p>
</td>
<td>
<p><span>0.947</span></p>
</td>
<td>
<p><span>0.057</span></p>
</td>
<td>
<p><span>0.961</span></p>
</td>
<td>
<p><span>0.939</span></p>
</td>
<td>
<p><span>0.950</span></p>
</td>
<td>
<p><span>0.045</span></p>
</td>
</tr>
<tr>
<td>
<p><span>Multilingual</span></p>
</td>
<td>
<p><span>0.901</span></p>
</td>
<td>
<p><span>0.899</span></p>
</td>
<td>
<p><span>0.900</span></p>
</td>
<td>
<p><span>0.054</span></p>
</td>
<td>
<p><span>0.906</span></p>
</td>
<td>
<p><span>0.892</span></p>
</td>
<td>
<p><span>0.899</span></p>
</td>
<td>
<p><span>0.051</span></p>
</td>
</tr>
<tr>
<td>
<p><span>Tool Use</span></p>
</td>
<td>
<p><span>0.884</span></p>
</td>
<td>
<p><span>0.958</span></p>
</td>
<td>
<p><span>0.920</span></p>
</td>
<td>
<p><span>0.126</span></p>
</td>
<td>
<p><span>0.876</span></p>
</td>
<td>
<p><span>0.946</span></p>
</td>
<td>
<p><span>0.909</span></p>
</td>
<td>
<p><span>0.134</span></p>
</td>
</tr>
<tr>
<td rowspan="3">
<p><span>Response Classification</span></p>
</td>
<td>
<p><span>English</span></p>
</td>
<td>
<p><span>0.947</span></p>
</td>
<td>
<p><span>0.931</span></p>
</td>
<td>
<p><span>0.939</span></p>
</td>
<td>
<p><span>0.040</span></p>
</td>
<td>
<p><span>0.947</span></p>
</td>
<td>
<p><span>0.925</span></p>
</td>
<td>
<p><span>0.936</span></p>
</td>
<td>
<p><span>0.040</span></p>
</td>
</tr>
<tr>
<td>
<p><span>Multilingual</span></p>
</td>
<td>
<p><span>0.929</span></p>
</td>
<td>
<p><span>0.805</span></p>
</td>
<td>
<p><span>0.862</span></p>
</td>
<td>
<p><span>0.033</span></p>
</td>
<td>
<p><span>0.931</span></p>
</td>
<td>
<p><span>0.785</span></p>
</td>
<td>
<p><span>0.851</span></p>
</td>
<td>
<p><span>0.031</span></p>
</td>
</tr>
<tr>
<td>
<p><span>Tool Use</span></p>
</td>
<td>
<p><span>0.774</span></p>
</td>
<td>
<p><span>0.884</span></p>
</td>
<td>
<p><span>0.825</span></p>
</td>
<td>
<p><span>0.176</span></p>
</td>
<td>
<p><span>0.793</span></p>
</td>
<td>
<p><span>0.865</span></p>
</td>
<td>
<p><span>0.827</span></p>
</td>
<td>
<p><span>0.155</span></p>
</td>
</tr>
</tbody>
</table>

# Get started

Llama Guard 3 is available by default on Llama 3.1 [reference implementations](https://github.com/meta-llama). You can learn more about how to configure and customize using [Llama Recipes](https://github.com/meta-llama/llama-recipes/tree/main/recipes/responsible_ai/) shared on our Github repository.

# Limitations
There are some limitations associated with Llama Guard 3. First, Llama Guard 3 itself is an LLM fine-tuned on Llama 3.1. Thus, its performance (e.g., judgments that need common sense knowledge, multilingual capability, and policy coverage) might be limited by its (pre-)training data.

Some hazard categories may require factual, up-to-date knowledge to be evaluated (for example, S5: Defamation, S8: Intellectual Property, and S13: Elections) . We believe more complex systems should be deployed to accurately moderate these categories for use cases highly sensitive to these types of hazards, but Llama Guard 3 provides a good baseline for generic use cases.

Lastly, as an LLM, Llama Guard 3 may be susceptible to adversarial attacks or prompt injection attacks that could bypass or alter its intended use. Please feel free to [report](https://github.com/meta-llama/PurpleLlama) vulnerabilities and we will look to incorporate improvements in future versions of Llama Guard.


# Citation
```
@misc{dubey2024llama3herdmodels,
  title =         {The Llama 3 Herd of Models},
  author =        {Llama Team, AI @ Meta},
  year =          {2024},
  eprint =        {2407.21783},
  archivePrefix = {arXiv},
  primaryClass =  {cs.AI},
  url =           {https://arxiv.org/abs/2407.21783}
}
```

# References

[1] [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)

[2] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)

[3] [Llama Guard 2 Model Card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL_CARD.md)

[4] [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviors in Large Language Models](https://arxiv.org/abs/2308.01263)