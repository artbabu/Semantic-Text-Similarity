# Semantic-Text-Similarity

## Introduction

In last five years, there has been an increase in the research on estimating the similarity be-
tween two sentence.It is a basic language understanding problems that is applicable in many
natural language processing application like machine Translation, Question answering etc,.
In distributional semantic models(DSM), the meaning of the words(representations) are ap-
proximated under a distributional vector space using the patterns of word co-occurring with
other words in the corpus. DSMs are insubstantial in modelling the sentence representations
as its doesn’t capture the logical words, grammatical and syntactical aspect [3].To encourage
research in this area, ACL holds SemEval competitions to evaluate the semantic models since
2012. Our goal in this proposal is to replicated some of below mentioned existing works to
get better understanding of these models.

## Task Overview

The task of semantic textual similarity involves two sub-task: 1)Sentence Relatedness, 2)
Sentence Entailment. In this project we focus on predicting the sentence relatedness for a
given sentence pair indicating how much the two sentence are related. For training and test of
the STS model, we will use data from SemEval-2017 which also includes the collection of STS
dataset released since 2012. This dataset consist of the sentence pairs with its corresponding
relatedness score.The scores takes value ranging from 0 indicating no similarity to 5 indicating
the high similarity between the sentences.
The performance of the model is measured using pearson correlation of the predicted
score with the human judgement score given in the dataset.
