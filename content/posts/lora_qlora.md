---
title: "LoRA and QLoRA: Elixir for the GPU poor"
date: 2025-01-15
# weight: 1
# aliases: ["/first"]
tags: ["llm","quantization"]
author: "Chhaya Kumar Das"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: true
hidemeta: false
comments: false
# description: "Desc Text."
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: true
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: false
UseHugoToc: false
# cover:
#     image: "<image path/url>" # image path/url
#     alt: "<alt text>" # alt text
#     caption: "<text>" # display caption under cover
#     relative: false # when using page bundles set this to true
#     hidden: true # only hide on current single page
# editPost:
#     URL: "https://github.com/<path_to_repo>/content"
#     Text: "Suggest Changes" # edit text
#     appendFilePath: true # to append file path to Edit link

---


# Need of the hour

Post ChatGPT and especially after the LLaMA release from Meta, everyone became fascinated with these big LLMs. But the early models were big. 60-70 billion parameter models were considered very light weight. Even if you had to finetune such '**light weight**' models you needed enormous compute. The community had to figure out an efficient way to finetune these models. This is where LoRA and QLoRA come to the rescue. Infact one can say that without these 2 techniques the open source llm landscape would be looking completely different.

# LoRA: Low Rank Adapters

LoRA stands for Low Rank Adapters. We will break this term in sometime. If we had to finetune something, the practice around that time was you have to update all the model weights. You can have few layers frozen but modify the weights of the remaining layers or add your layers and train them.  What do I mean by this? Lets say I have a 7 billion parameter model with 28 layers. When I finetune for my dataset either all or some of these layers have to finetuned. Now when your model is very big, even a single layer can boast an enormous amount of parameters. Plus if I finetune because I am messing with the already trained weights my model may give worse output. The researchers at Microsoft thought that what if we can have some way that adds an extra set of weights beside the already existing weights? We only train the added weights and keep the pretrained weights frozen. This is the base idea of LoRA

## How it works?

