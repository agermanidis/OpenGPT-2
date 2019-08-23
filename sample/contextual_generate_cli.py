import tensorflow as tf
import numpy as np
import sys
import json
import sys

sys.path.append('../')
from lm.modeling import GroverModel, GroverConfig, _top_p_sample, sample
from sample.encoder import get_encoder, format_context, _tokenize_article_pieces, extract_generated_target
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Contextual generation (aka given some metadata we will generate articles')
parser.add_argument(
    '-metadata_fn',
    dest='metadata_fn',
    type=str,
    help='Path to a JSONL containing metadata',
)
parser.add_argument(
    '-out_fn',
    dest='out_fn',
    type=str,
    help='Out jsonl, which will contain the completed jsons',
)
parser.add_argument(
    '-input',
    dest='input',
    type=str,
    help='Text to complete',
)
parser.add_argument(
    '-model_config_fn',
    dest='model_config_fn',
    default='../lm/configs/base.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '-model_ckpt',
    dest='model_ckpt',
    default='../models/base/model.ckpt',
    type=str,
    help='checkpoint file for the model',
)
parser.add_argument(
    '-target',
    dest='target',
    default='article',
    type=str,
    help='What to generate for each item in metadata_fn. can be article (body), title, etc.',
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    default=1,
    type=int,
    help='How many things to generate per context. will split into chunks if need be',
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds. useful if we want to split up a big file into multiple jobs.',
)
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on. useful if we want to split up a big file into multiple jobs.'
)
parser.add_argument(
    '-max_batch_size',
    dest='max_batch_size',
    default=None,
    type=int,
    help='max batch size. You can leave this out and we will infer one based on the number of hidden layers',
)
parser.add_argument(
    '-top_p',
    dest='top_p',
    default=0.95,
    type=float,
    help='p to use for top p sampling. if this isn\'t none, use this for everthing'
)
parser.add_argument(
    '-samples',
    dest='samples',
    default=1,
    type=int,
    help='num_samples',
)

args = parser.parse_args()

encoder = get_encoder()
news_config = GroverConfig.from_json_file(args.model_config_fn)

# We might have to split the batch into multiple chunks if the batch size is too large
default_mbs = {12: 32, 24: 16, 48: 3}
max_batch_size = args.max_batch_size if args.max_batch_size is not None else default_mbs[news_config.num_hidden_layers]

# factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
num_chunks = int(np.ceil(args.batch_size / max_batch_size))
batch_size_per_chunk = int(np.ceil(args.batch_size / num_chunks))
print("\n~~\nbatch size={}, max batch size={}, num chunks={}, batch size per chunk={}\n~~\n".format(
    args.batch_size, max_batch_size, num_chunks, batch_size_per_chunk), flush=True)

# This controls the top p for each generation.
top_p = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * args.top_p

# with open(args.metadata_fn, 'r') as f:
#     articles = [json.loads(l) for i, l in enumerate(f) if i % args.num_folds == args.fold]

tf_config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
    p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
    eos_token = tf.placeholder(tf.int32, [])
    tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                           eos_token=eos_token, ignore_ids=None, p_for_topp=p_for_topp,
                           do_topk=False)

    saver = tf.train.Saver()
    saver.restore(sess, args.model_ckpt)
    print('Loaded model.')
    text = input()
    while text != "":
        for i in range(args.samples):
            print("Sample,", i + 1, " of ", args.samples)
            # Let's go!
            encoded = _tokenize_article_pieces(encoder, text)
            context_formatted = []
            # for key in ['domain', 'date', 'authors', 'title', 'article']:
            #     if key != args.target:
            #         context_formatted.extend(article_pieces.pop(key, []))
            # context_formatted.append(encoder.__dict__['begin_{}'.format(args.target)])
            context_formatted.extend(encoded[:-1])
            # Format context end

            # Indices we definitely DONT WANT TO PREDICT
            ignore_ids_np = np.array(encoder.special_tokens_onehot)
            ignore_ids_np[encoder.endoftext] = 0

            gens = []
            gens_raw = []
            gen_probs = []

            # article['top_ps'] = top_p.reshape(-1).tolist()
            for chunk_i in range(num_chunks):
                tokens_out, probs_out = sess.run([tokens, probs],
                                                 feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                                                            eos_token: 60000,
                                                            p_for_topp: top_p[chunk_i]})

                for t_i, p_i in zip(tokens_out, probs_out):
                    extraction = extract_generated_target(output_tokens=t_i, encoder=encoder, target=args.target)
                    gens.append(extraction['extraction'])

            # article['gens_{}'.format(args.target)] = gens
            # article['gensraw_{}'.format(args.target)] = gens_raw
            # article['probs_{}'.format(args.target)] = gen_probs

            # these were in there for whatever reason...
            # article.pop('input_ids_conditional', None)
            # article.pop('input_ids_unconditional', None)
            # f_out.write(json.dumps(article) + '\n')
            print(gens[0])
        text = input()

