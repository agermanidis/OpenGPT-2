#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

import runway


sess = None

@runway.setup(options={'checkpoint_dir': runway.file(is_directory=True)})
def setup(opts):
    global sess
    global output
    global enc
    length=None
    temperature=1
    top_k=0

    enc = encoder.get_encoder(opts['checkpoint_dir'])
    hparams = model.default_hparams()
    with open(os.path.join(opts['checkpoint_dir'], 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    sess = tf.Session()
    context = tf.placeholder(tf.int32, [1, None])
    output = sample.sample_sequence(
        hparams=hparams, 
        length=length,
        context=context,
        batch_size=1,
        temperature=temperature,
        top_k=top_k
    )
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(opts['checkpoint_dir'])
    saver.restore(sess, ckpt)

    return sess, enc, context


@runway.command('generate', inputs={'prompt': runway.text, 'seed': runway.number(default=0, max=999)}, outputs={'text': runway.text})
def generate(model, inputs):
    sess, enc, context = model
    seed = inputs['seed']
    np.random.seed(seed)
    tf.set_random_seed(seed)
    context_tokens = enc.encode(inputs['prompt'])
    out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
    result = enc.decode(out[0])
    return result


if __name__ == '__main__':
    runway.run()
 
