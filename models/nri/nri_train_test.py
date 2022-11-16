import os
import time

import torch
from torch.optim import lr_scheduler
from torch import optim

from models.nri.sourcecode.modules import *
from models.nri.sourcecode.utils import *

def concatenate_tensor_in_one_list(t: torch.Tensor, to_list):
    t_ = []
    for t_batch in t:
        for t_sample in t_batch.cpu().detach().numpy():
            t_.append(t_sample.tolist() if to_list else t_sample)

    return t_ if to_list else np.array(t_)

def aggregate_results(connection_graph):
    coef_mat_est = []

    # TODO: zavrsi samo ovo jos
    a = np.array([
        [[2, 1], [1, 3]],
        [[3, 2], [3, 2]],
    ])

    a = np.argmax(a, axis=-1)
    print(a)

    a = np.median(a, axis=0)
    print(a)

    a[a > 0.] = 1
    print(a)

    n_data = int((1 + np.sqrt(1 + 4 * len(a))) / 2)
    print(n_data)
    # END TODO

    return coef_mat_est

def train(epoch, best_val_loss, train_loader, valid_loader, config):
    """
    Performs one train loop and one test loop
    epoch: number of epoch
    best_val_loss: best validation loss so far
    train_loader: loader with train data
    valid_loader: loader with test data
    :returns validation accuracy
    """
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    scheduler.step()

    # region TRAIN_LOOP
    for batch_idx, (data, relations) in enumerate(train_loader):

        if config.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)

        optimizer.zero_grad()

        logits = encoder(data, rel_rec, rel_send)
        #bs = logits.shape[0]
        #logits = logits.mean(axis=0).unsqueeze(axis=0).repeat(bs, 1, 1)
        edges = gumbel_softmax(logits, tau=config.temp, hard=config.hard)
        #edges = torch.tensor(np.tile(np.array([[1., 0.], [0., 1.]]), (edges.size(0), 1, 1)), device='cuda:0')
        prob = my_softmax(logits, -1)

        if config.decoder == 'rnn':
            output = decoder(data, edges, rel_rec, rel_send, 100,
                             burn_in=True,
                             burn_in_steps=config.timesteps - config.prediction_steps)
        else:
            output = decoder(data, edges, rel_rec, rel_send,
                             config.prediction_steps)

        target = data[:, :, 1:, :]

        loss_nll = nll_gaussian(output, target, config.var)

        if config.prior:
            loss_kl = kl_categorical(prob, log_prior, config.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, config.num_atoms,
                                             config.edge_types)

        loss = loss_nll + loss_kl

        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)

        loss.backward()
        optimizer.step()

        #mse = F.mse_loss(output, target)
        mse_train.append(F.mse_loss(output, target).item()) # B: .data[0]
        nll_train.append(loss_nll.item()) # B: .data[0]
        kl_train.append(loss_kl.item()) # B: .data[0]
    # endregion

    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    encoder.eval()
    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if config.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        logits = encoder(data, rel_rec, rel_send)
        #bs = logits.shape[0]
        #logits = logits.mean(axis=0).unsqueeze(axis=0).repeat(bs, 1, 1)
        edges = gumbel_softmax(logits, tau=config.temp, hard=True)
        #edges = torch.tensor(np.tile(np.array([[1., 0.], [0., 1.]]), (edges.size(0), 1, 1)), device='cuda:0')

        prob = my_softmax(logits, -1)

        # validation output uses teacher forcing
        output = decoder(data, edges, rel_rec, rel_send, 1)

        target = data[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, config.var)
        loss_kl = kl_categorical_uniform(prob, config.num_atoms, config.edge_types)

        acc = edge_accuracy(logits, relations)
        acc_val.append(acc)

        mse_val.append(F.mse_loss(output, target).item()) # B: .data[0]
        nll_val.append(loss_nll.item()) # B: .data[0]
        kl_val.append(loss_kl.item()) # B: .data[0]

    if config.verbose:
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
    if config.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        if config.verbose:
            print('Best model so far, saving...')

    # Save results in result-scraping dict object
    results['train']['time'].append(time.time()-t)
    results['train']['nll'].append(np.mean(nll_train))
    results['train']['kl'].append(np.mean(kl_train))
    results['train']['mse'].append(np.mean(mse_train))
    results['train']['acc'].append(np.mean(acc_train))
    results['valid']['nll'].append(np.mean(nll_val))
    results['valid']['kl'].append(np.mean(kl_val))
    results['valid']['mse'].append(np.mean(mse_val))
    results['valid']['acc'].append(np.mean(acc_val))

    return np.mean(nll_val)

def test(test_loader, config):
    acc_test = []
    nll_test = []
    kl_test = []
    mse_test = []
    tot_mse = 0
    counter = 0

    connection_graph = []
    test_predictions = []
    test_targets = []

    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))
    for batch_idx, (data, relations) in enumerate(test_loader):
        if config.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        assert (data.size(2) - config.timesteps) >= config.timesteps

        data_encoder = data[:, :, :config.timesteps, :].contiguous()
        data_decoder = data[:, :, -config.timesteps:, :].contiguous()

        logits = encoder(data_encoder, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=config.temp, hard=True)

        prob = my_softmax(logits, -1)
        connection_graph.append(prob)

        output = decoder(data_decoder, edges, rel_rec, rel_send, 1)

        target = data_decoder[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, config.var)
        loss_kl = kl_categorical_uniform(prob, config.num_atoms, config.edge_types)

        acc = edge_accuracy(logits, relations)
        acc_test.append(acc)

        mse_test.append(F.mse_loss(output, target).item())
        nll_test.append(loss_nll.item())
        kl_test.append(loss_kl.item())

        # For plotting purposes
        if config.decoder == 'rnn':
            if config.dynamic_graph:
                output = decoder(data, edges, rel_rec, rel_send, 100,
                                 burn_in=True, burn_in_steps=config.timesteps,
                                 dynamic_graph=True, encoder=encoder,
                                 temp=config.temp)
            else:
                output = decoder(data, edges, rel_rec, rel_send, 100,
                                 burn_in=True, burn_in_steps=config.timesteps)
            output = output[:, :, config.timesteps:, :]
            test_predictions.append(output)

            target = data[:, :, -config.timesteps:, :]
            test_targets.append(target)
        else:
            data_plot = data[:, :, config.timesteps:config.timesteps + 21,
                        :].contiguous()
            output = decoder(data_plot, edges, rel_rec, rel_send, 20)
            test_predictions.append(output)

            target = data_plot[:, :, 1:, :]
            test_targets.append(target)

        mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
        tot_mse += mse.data.cpu().numpy()
        counter += 1

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    if config.verbose:
        print('--------------------------------')
        print('--------Testing-----------------')
        print('--------------------------------')
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)))
        print('MSE: {}'.format(mse_str))

    results['test']['nll'].append(np.mean(nll_test))
    results['test']['kl'].append(np.mean(kl_test))
    results['test']['mse'].append(np.mean(mse_test))
    results['test']['acc'].append(np.mean(acc_test))

    # Concatenate results of all batches (and save the prediction/targets from the decoder)
    connection_graph = concatenate_tensor_in_one_list(connection_graph, to_list=False)
    test_predictions = concatenate_tensor_in_one_list(test_predictions, to_list=True)
    test_targets = concatenate_tensor_in_one_list(test_targets, to_list=True)

    results['test']['predictions'] = test_predictions
    results['test']['targets'] = test_targets

    return connection_graph


def train_test(config, train_loader, valid_loader, test_loader):
    global encoder, decoder, optimizer, scheduler, rel_rec, rel_send, triu_indices, tril_indices, log_prior, encoder_file, decoder_file, results

    config.cuda = torch.cuda.is_available()
    config.factor = not config.no_factor
    if config.dynamic_graph and config.verbose:
        print("Testing with dynamically re-computed graph.")

    # Generate off-diagonal interaction graph
    off_diag = np.ones([config.num_atoms, config.num_atoms]) - np.eye(config.num_atoms)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)

    # Setup encoder/decoder architectures
    if config.encoder == 'mlp':
        encoder = MLPEncoder(config.timesteps * config.dims, config.encoder_hidden,
                             config.edge_types,
                             config.encoder_dropout, config.factor)
    elif config.encoder == 'cnn':
        encoder = CNNEncoder(config.dims, config.encoder_hidden,
                             config.edge_types,
                             config.encoder_dropout, config.factor)
    if config.decoder == 'mlp':
        decoder = MLPDecoder(n_in_node=config.dims,
                             edge_types=config.edge_types,
                             msg_hid=config.decoder_hidden,
                             msg_out=config.decoder_hidden,
                             n_hid=config.decoder_hidden,
                             do_prob=config.decoder_dropout,
                             skip_first=config.skip_first)
    elif config.decoder == 'rnn':
        decoder = RNNDecoder(n_in_node=config.dims,
                             edge_types=config.edge_types,
                             n_hid=config.decoder_hidden,
                             do_prob=config.decoder_dropout,
                             skip_first=config.skip_first)
    elif config.decoder == 'sim':
        decoder = SimulationDecoder(config.loc_max, config.loc_min, config.vel_max, config.vel_min, config.suffix)
    encoder_file = os.path.join(config.load_folder, 'encoder.pt')
    decoder_file = os.path.join(config.load_folder, 'decoder.pt')

    # Set up optimizer and scheduler
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay,
                                    gamma=config.gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(config.num_atoms)
    tril_indices = get_tril_offdiag_indices(config.num_atoms)

    # Set up prior
    if config.prior:
        prior = np.array([0.95, 0.05])  # , 0.03, 0.03])
        print("Using prior")
        print(prior)
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = Variable(log_prior)
        if config.cuda:
            log_prior = log_prior.cuda()

    # Put everything to cuda
    if config.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    # Train loop
    results = {
        'train': {
            'nll': [],
            'kl': [],
            'mse': [],
            'acc': [],
            'time': []
        },
        'valid': {
            'nll': [],
            'kl': [],
            'mse': [],
            'acc': []
        },
        'test': {
            'nll': [],
            'kl': [],
            'mse': [],
            'acc': [],
            'predictions': [],
            'targets': []
        }
    }
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(config.epochs):
        val_loss = train(epoch, best_val_loss, train_loader, valid_loader, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    if config.verbose:
        print("Optimization finished")
        print(f"Best epoch {best_epoch}")
    connection_graph = test(test_loader, config)

    # TODO: Get somehow coef_mat_est from the outputs inferred on test set
    coef_mat_est = aggregate_results(connection_graph)

    return coef_mat_est, results
