# coding=utf-8
import torch
import numpy as np
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamSearchScorer,
)


def generate_beam(decoder, tokenizer, length, embed, device: str = 'cuda', beam_size: int = 1,
                  return_top_pred: bool = False,
                  entry_length=20, temperature=1.0, stop_token_index: int = 50256):
    generated = embed
    tokens = None
    scores = None
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    scores_list = None

    for _ in range(entry_length):
        outputs = decoder(inputs_embeds=generated)
        # 一个单词在所有层上的概率之和
        logits = outputs.logits  # ?
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.softmax(-1).log()
        # Find the largest possible word for that location
        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)  # return max score and index
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            scores_list = scores.unsqueeze(-1)  # Each array holds the maximum probability of all words
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
        else:

            logits[is_stopped] = torch.tensor(-float(np.inf)).to(device)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits  # qian scores

            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            z = scores_sum_average.view(-1)
            scores_sum_average, next_tokens = z.topk(
                beam_size, -1
            )
            next_tokens_source = next_tokens // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            scores_list = torch.cat(
                [scores_list[next_tokens_source], logits[next_tokens_source, next_tokens].unsqueeze(-1)], dim=-1)
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)  #####
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]
        z = next_tokens.squeeze()
        next_token_embed = decoder.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
        if is_stopped.all():
            break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length) - 1])  # .strip()
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    # if return_top_pred:
    #    tokens = torch.tensor(tokenizer(output_texts[0])['input_ids'], dtype=torch.int64)
    #    scores = scores_list[0]
    #    return tokens.to(device),-torch.mean(scores[:,:tokens.shape[1]])
    print(output_texts)
    token = []
    token.append(torch.tensor(tokenizer(output_texts[0])['input_ids'], dtype=torch.int64).to(device))
    scores = -torch.mean(scores_list[0, :tokens.shape[1]]).unsqueeze(0)
    for i in range(1, beam_size):
        token.append(torch.tensor(tokenizer(output_texts[i])['input_ids'], dtype=torch.int64).to(device))
        scores = torch.cat((scores, -torch.mean(scores_list[i, :tokens.shape[1]]).unsqueeze(0)), dim=0)

    return token, scores  # Returns the probability of this sentence

def trainer(decoder:torch.nn.Module = None,
            prefix: torch.LongTensor = None,
            question: torch.LongTensor = None,
            answer: torch.LongTensor = None,
            explanation: torch.LongTensor = None,
            q_a_ids: torch.LongTensor =None,
            q_e_ids: torch.LongTensor = None,
            q_e_a_ids: torch.LongTensor = None,
            q_a_sigment: torch.LongTensor = None,
            q_e_sigment: torch.LongTensor = None,
            q_e_a_sigment: torch.LongTensor = None,
            q_e_a_len = None,
            q_a_len = None,
            q_e_len = None,
            len_qea = None,
            num_beam: int =3,
            device: str = 'cpu',
            tokenizer = None,
            label_q_a=None,
            label_q_e_a=None,
            label_q_e=None,
            text_qc = None,
            text_ans = None,
            ):

    # text pre
    input_ids = text_qc.data['input_ids']

    #attention_mask = torch.tensor(text_qc.data['attention_mask'],dtype=torch.long).to(device)

    out_a = decoder(
        input_ids = q_a_ids,
        encoder_hidden_states=prefix,
        return_dict=True,
        labels=label_q_a
    )
    out_e = decoder(
        input_ids = q_e_ids,
        encoder_hidden_states=prefix,
        return_dict=True,
        labels = label_q_e
    )


    #   prepare for beam search
    score = []
    text = []
    text_label = []
    start = []
    for j in range(num_beam):
        text_beam = []
        text_label_beam = []
        start_beam = []
        text.append(text_beam)
        text_label.append(text_label_beam)
        start.append(start_beam)
    for i in range(prefix.shape[0]):
        input_id = torch.tensor(input_ids[i], dtype=torch.long).to(device)
        ones = torch.ones((num_beam, 1), dtype=torch.long).to(device)
        input_ids_true = ones * input_id
        model_kwargs = {
            "encoder_hidden_states": prefix[i].unsqueeze(0),
        }
        beam_scorer_context = BeamSearchScorer(
            batch_size=1,
            num_beams=num_beam,
            device=device,
            do_early_stopping=True,
            num_beam_hyps_to_keep=num_beam,
            max_length=60
        )
        stopping_criteria_context = StoppingCriteriaList([MaxLengthCriteria(max_length=30)])
        logits_processor_context = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(3, eos_token_id=decoder.module.config.eos_token_id),
            ]
        )
        outputs = decoder.module.beam_search(input_ids_true, beam_scorer_context, logits_processor=logits_processor_context,output_scores=True,
                                        return_dict_in_generate = True,
                                        stopping_criteria=stopping_criteria_context, pad_token_id=50257, **model_kwargs)
        scores = - outputs.sequences_scores

        score.append(scores.unsqueeze(0))
        outputs = outputs.sequences
        outputs = tokenizer.batch_decode(outputs,skip_special_tokens=True)
        #print(outputs)
        for j in range(num_beam):
            text_q_c_a = tokenizer(outputs[j])['input_ids']+text_ans[i]
            start[j].append((len(tokenizer(outputs[j])['input_ids'])+2))
            text_label[j].append([-100]*(len(tokenizer(outputs[j])['input_ids'])+2)+text_ans[i][2:]+[-100]*(50-len(text_q_c_a)))
            text[j].append(text_q_c_a+[tokenizer.pad_token_id]*(50-len(text_q_c_a)))
    score = torch.cat(score,dim=0).to(device)
    score = score.T
    loss = 0
    text = torch.tensor(text,dtype=torch.long).to(device)
    text_label = torch.tensor(text_label,dtype=torch.long).to(device)
    loss_r = torch.zeros(1,dtype=torch.long).to(device)
    for beam in range(num_beam):

        output = decoder(
            input_ids=text[beam],
            # token_type_ids = q_e_sigment,
            encoder_hidden_states=prefix,
            return_dict=True,
            labels=text_label[beam]
        )
        out_r = output.logits
        loss = loss + output.loss
        reward = torch.zeros(len(out_r)).cuda()
        for bs in range(len(out_r)):
            gt_score = 0.
            wei_score = 0.
            for i, j, k in zip(out_a.logits[bs][(q_a_len[bs]-len(text_ans[bs][2:])):q_a_len[bs]], text_ans[bs][2:], out_r[bs][start[beam][bs]:start[beam][bs]+len(text_ans[bs][2:])] ):
                gt_score += i[j]
                wei_score += k[j]
            gt_score = gt_score / len(text_ans[bs][2:])
            wei_score = wei_score / len(text_ans[bs][2:])
            reward[bs] = wei_score - gt_score
        loss_r = torch.mean(torch.clamp(reward * score[beam],min=-10.0)) + loss_r
    reward_loss = loss_r /num_beam

    return out_a.loss+out_e.loss +reward_loss + loss/num_beam





