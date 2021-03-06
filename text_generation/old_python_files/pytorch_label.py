from skimage.io import imread
from skimage.transform import resize
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from pytorch_tokenizer import create_tokenizer
from torch.distributions import Categorical
from utils import decode_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def temperature_sampling(encoder, decoder, tokenizer, image_paths=None, 
                        images=None, temperature=1.5, max_len=100, rl=False):
    # continue only image_path XOR image is given
    assert (
        not(image_paths is None and images is None) and 
        not(image_paths is not None and images is not None)
    )

    vocab_size = len(tokenizer)

    if image_paths is not None:
        images = []

        for image_path in image_paths:
            # Read image and process
            img = imread(image_path)
            img = resize(img, (256, 256, 3))
            images.append(img)

        images = torch.FloatTensor(images).to(device) # (len(image_paths), 256, 256, 3)
        images = images.permute(0, 3, 1, 2) # (len(image_paths), 3, 256, 256)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        images = transform(images) # (len(image_paths), 3, 256, 256)
        images = images.permute(0, 2, 3, 1) # (len(image_paths), 256, 256, 3)

    # Encode
    batch_size = images.shape[0]
    # image = images.unsqueeze(0)  # (batch_size, 256, 256, 3)
    images = images.to(device)
    encoder_out = encoder(images)  # (batch_size, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # Tensors for generating
    prev_words = torch.LongTensor([[tokenizer.stoi['<startseq>']]] * batch_size).to(device) # (batch_size, 1)
    
    # Variables for keeping track of
    dones = [False for _ in range(batch_size)]
    step = 1

    # Init hidden states
    h, c = decoder.init_hidden_state(encoder_out) # (batch_size, decoder_dim)

    # Generated sequences
    seqs = [[] for _ in range(batch_size)]

    # Saved chosen action value and its log prob
    decoder.saved_actions = [[] for _ in range(batch_size)]

    while True:
        # Shared layers
        embeddings = decoder.embedding(prev_words).squeeze(1)  # (batch_size, embed_dim)
        
        awe, alpha = decoder.attention(encoder_out, h)  # (batch_size, encoder_dim), (batch_size, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (batch_size, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (batch_size, encoder_dim)
        awe = gate * awe

        decoder_input = torch.cat([embeddings, awe], dim=1)
        h, c = decoder.decode_step(decoder_input, (h, c))  # (batch_size, decoder_dim)

        # Probs
        scores = decoder.fc(h)  # (batch_size, vocab_size)
        scores = F.softmax(scores / temperature, dim=1) # (batch_size, vocab_size)

        try:
            dist = Categorical(scores)
        except ValueError as e:
            print(f"{scores=}")
            print(e)
        indices = dist.sample() # (batch_size, vocab_size)
        
        for i, (seq, idx) in enumerate(zip(seqs, indices)):
            if not dones[i]:
                seq.append(idx.item())

        if rl:
            values = decoder.v(h) # (batch_size, 1)
            log_probs = dist.log_prob(indices) # (batch_size)

            # Why torch.squeeze yields dim = 0 ????
            if torch.squeeze(values).dim() > 0:
                zipped = zip(log_probs, torch.squeeze(values))
            else:
                zipped = zip(log_probs, values)
            
            for i, (log_prob, value) in enumerate(zipped):
                if not dones[i]:
                    decoder.saved_actions[i].append([log_prob, value])

        # Finished?
        for i in range(batch_size):
            idx = indices[i].item()
            if idx == tokenizer.stoi['<endseq>']:
                dones[i] = True

        if all(dones) or step > max_len:
            # print(f"Finished! With lengths: {[len(e) for e in decoder.saved_actions]}")
            # print(f"Seq lengths: {[len(e) for e in seqs]}")
            break

        prev_words = torch.unsqueeze(indices, 1)
        step += 1
    
    return seqs, None

def caption_image_beam_search(encoder, decoder, tokenizer, image_path=None, image=None, beam_size=3):
    """
    Reads an image and captions it with beam search.
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(tokenizer)

    if image_path is not None:
        # Read image and process
        img = imread(image_path)
        img = resize(img, (256, 256, 3))

        img = torch.FloatTensor(img).to(device)
        img = img.permute(2, 0, 1) # (3, 256, 256)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)  # (3, 256, 256)
        image = image.permute(1, 2, 0) # (256, 256, 3)

    # Encode
    image = image.unsqueeze(0)  # (1, 256, 256, 3)
    image = image.to(device)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[tokenizer.stoi['<startseq>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        # print(step)
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        # print(prev_word_inds)
        # print(next_word_inds)
        # print(top_k_words)
        # print(top_k_scores)
        # print(seqs)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != tokenizer.stoi['<endseq>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

def test_beam_search():
    import time
    tokenizer = create_tokenizer()
    checkpoint_path = 'weights/pytorch_attention/checkpoint_2021-12-14_13-13-34.062632.pth.tar'
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder.to(device)
    decoder.to(device)
    image_path = 'data/images/images_normalized/3095_IM-1448-1001.dcm.png'
    start = time.time()
    seq, _ = caption_image_beam_search(encoder, decoder, image_path, tokenizer, beam_size=5)
    end = time.time()
    print(f"time taken = {end-start:.3f} seconds")
    print([tokenizer.itos[idx] for idx in seq])

def test_temperature_sampling():
    import time
    tokenizer = create_tokenizer()
    # checkpoint_path = 'weights/pytorch_attention/checkpoint_2022-01-07_17-44-48.868114.pth.tar' # RL
    checkpoint_path = "weights\pytorch_attention\checkpoint_2021-12-21_03-32-00.004925.pth.tar" # Normal
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder.to(device)
    decoder.to(device)
    image_paths = [
        'data/images/images_normalized/3095_IM-1448-1001.dcm.png',
        'data/images/images_normalized/1_IM-0001-3001.dcm.png',
    ]
    start = time.time()
    seqs, _ = temperature_sampling(encoder, decoder, tokenizer, image_paths=image_paths, temperature=1.2, max_len=100)
    end = time.time()
    print(seqs)
    print(f"time taken = {end-start:.3f} seconds")
    print(f"time taken per image = {(end-start)/len(image_paths):.3f} seconds")
    # print([' '.join([tokenizer.itos[idx] for idx in seq]) for seq in seqs])
    print(decode_sequences(tokenizer, seqs))

if __name__ == "__main__":
    test_temperature_sampling()