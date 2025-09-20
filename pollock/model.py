import logging
import random
import time
from collections import Counter

import anndata
import numpy as np
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

class ZINBLoss(torch.nn.Module):
    """
    Adapted from scDCC https://github.com/ttgump/scDCC/blob/65bcbbd63e2e80785a3f4d9bd8f3cedd8f38f6ca/layers.py
    """
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class MeanAct(torch.nn.Module):
    """
    Pulled from scDCC https://github.com/ttgump/scDCC/blob/65bcbbd63e2e80785a3f4d9bd8f3cedd8f38f6ca/layers.py
    """
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(torch.nn.Module):
    """
    Pulled from scDCC https://github.com/ttgump/scDCC/blob/65bcbbd63e2e80785a3f4d9bd8f3cedd8f38f6ca/layers.py
    """
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class PollockModel(torch.nn.Module):
    def __init__(self, genes, classes,
                 latent_dim=64, enc_out_dim=128, middle_dim=512,
                 zinb_scaler=1., kl_scaler=1e-5, clf_scaler=1.):
        """
        Pollock VAE + classifier
        """
        super(PollockModel, self).__init__()
        self.latent_dim = latent_dim
        self.genes = genes
        self.n_genes = len(genes)
        self.classes = classes
        self.n_classes = len(classes)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_genes, middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim, enc_out_dim),
            torch.nn.ReLU(),
        )

        self.mu = torch.nn.Linear(enc_out_dim, latent_dim)
        self.var = torch.nn.Linear(enc_out_dim, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, enc_out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(enc_out_dim, middle_dim),
            torch.nn.ReLU(),
        )
        self.disp_decoder = torch.nn.Sequential(
            torch.nn.Linear(middle_dim, self.n_genes),
            DispAct()
        )
        self.mean_decoder = torch.nn.Sequential(
            torch.nn.Linear(middle_dim, self.n_genes),
            MeanAct()
        )
        self.drop_decoder = torch.nn.Sequential(
            torch.nn.Linear(middle_dim, self.n_genes),
            torch.nn.Sigmoid()
        )

        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, self.n_classes),
            torch.nn.Softmax(dim=1),
        )

        self.zinb_loss = ZINBLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.zinb_scaler = zinb_scaler
        self.kl_scaler = kl_scaler
        self.clf_scaler = clf_scaler

    def kl_divergence(self, z, mu, std):
        # lightning imp.
        # Monte carlo KL divergence
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)

        return kl

    def encode(self, x, use_means=False):
        x_encoded = self.encoder(x)
        mu, log_var = self.mu(x_encoded), self.var(x_encoded)

        # sample z from parameterized distributions
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        # get our latent
        if use_means:
            z = mu
        else:
            z = q.rsample()

        return z, mu, std

    def decode(self, x):
        h = self.decoder(x)
        x_disp = self.disp_decoder(h)
        x_mean = self.mean_decoder(h)
        x_drop = self.drop_decoder(h)

        return x_disp, x_mean, x_drop

    def calculate_loss(self, r, x_raw, scale_factor, y_true):
        reconstruction_loss = self.zinb_loss(
            x_raw, r['x_mean'], r['x_disp'], r['x_drop'], scale_factor=scale_factor)

        kl_loss = torch.mean(self.kl_divergence(r['z'], r['mu'], r['std']))

        clf_loss = torch.mean(self.ce_loss(r['y'], y_true))

        return ((reconstruction_loss * self.zinb_scaler) + (kl_loss * self.kl_scaler) + (clf_loss * self.clf_scaler),
                reconstruction_loss,
                kl_loss,
                clf_loss)

    def forward(self, x, use_means=False):
        z, mu, std = self.encode(x, use_means=use_means)
        x_disp, x_mean, x_drop = self.decode(z)
        y = self.prediction_head(z)

        return {
            'z': z,
            'mu': mu,
            'std': std,
            'x_disp': x_disp,
            'x_mean': x_mean,
            'x_drop': x_drop,
            'y': y
        }


def fit_model(model, opt, scheduler, train_dl, val_dl, epochs=20):
    """
    Enhanced fit_model function with comprehensive accuracy metrics.
    """
    use_cuda = next(model.parameters()).is_cuda
    history = []
    
    # Get class names for detailed reporting
    class_names = model.classes if hasattr(model, 'classes') else None
    n_classes = len(class_names) if class_names else model.n_classes
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # ================================
        # TRAINING PHASE
        # ================================
        model.train()
        train_total_loss = 0.
        train_recon_loss = 0.
        train_kl_loss = 0.
        train_clf_loss = 0.
        train_batches = 0
        
        # For training accuracy calculation
        train_predictions = []
        train_true_labels = []
        
        for i, b in enumerate(train_dl):
            x, x_raw, sf, y = b['x'], b['x_raw'], b['size_factor'], b['y']
            if use_cuda:
                x, x_raw, sf, y = x.cuda(), x_raw.cuda(), sf.cuda(), y.cuda()
            
            opt.zero_grad()
            out = model(x)
            total_loss, recon_loss, kl_loss, clf_loss = model.calculate_loss(out, x_raw, sf, y)
            total_loss.backward()
            opt.step()
            
            # Accumulate training losses
            train_total_loss += float(total_loss.detach().cpu())
            train_recon_loss += float(recon_loss.detach().cpu())
            train_kl_loss += float(kl_loss.detach().cpu())
            train_clf_loss += float(clf_loss.detach().cpu())
            train_batches += 1
            
            # Collect predictions for accuracy calculation
            _, predicted = torch.max(out['y'], 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_true_labels.extend(y.cpu().numpy())
            
            scheduler.step()
        
        # Calculate training metrics
        train_total_loss /= train_batches
        train_recon_loss /= train_batches
        train_kl_loss /= train_batches
        train_clf_loss /= train_batches
        
        # Calculate training accuracy metrics
        train_accuracy = accuracy_score(train_true_labels, train_predictions) * 100
        train_macro_f1 = f1_score(train_true_labels, train_predictions, average='macro') * 100
        train_weighted_f1 = f1_score(train_true_labels, train_predictions, average='weighted') * 100
        
        # ================================
        # VALIDATION PHASE
        # ================================
        model.eval()
        val_total_loss = 0.
        val_recon_loss = 0.
        val_kl_loss = 0.
        val_clf_loss = 0.
        val_batches = 0
        
        # For validation accuracy calculation
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for i, b in enumerate(val_dl):
                x, x_raw, sf, y = b['x'], b['x_raw'], b['size_factor'], b['y']
                if use_cuda:
                    x, x_raw, sf, y = x.cuda(), x_raw.cuda(), sf.cuda(), y.cuda()
                
                out = model(x)
                total_loss, recon_loss, kl_loss, clf_loss = model.calculate_loss(out, x_raw, sf, y)
                
                val_total_loss += float(total_loss.detach().cpu())
                val_recon_loss += float(recon_loss.detach().cpu())
                val_kl_loss += float(kl_loss.detach().cpu())
                val_clf_loss += float(clf_loss.detach().cpu())
                val_batches += 1
                
                # Collect predictions for accuracy calculation
                _, predicted = torch.max(out['y'], 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(y.cpu().numpy())
        
        # Calculate validation metrics
        val_total_loss /= val_batches
        val_recon_loss /= val_batches
        val_kl_loss /= val_batches
        val_clf_loss /= val_batches
        
        # Calculate validation accuracy metrics
        val_accuracy = accuracy_score(val_true_labels, val_predictions) * 100
        val_macro_f1 = f1_score(val_true_labels, val_predictions, average='macro') * 100
        val_weighted_f1 = f1_score(val_true_labels, val_predictions, average='weighted') * 100
        
        # Calculate per-class metrics for validation (optional detailed analysis)
        try:
            val_class_report = classification_report(val_true_labels, val_predictions, 
                                                   target_names=class_names, 
                                                   output_dict=True, zero_division=0)
            # Extract per-class F1 scores
            per_class_f1 = {}
            if class_names:
                for class_name in class_names:
                    if class_name in val_class_report:
                        per_class_f1[f'val_f1_{class_name}'] = val_class_report[class_name]['f1-score'] * 100
        except:
            val_class_report = None
            per_class_f1 = {}
        
        epoch_time = time.time() - start_time
        
        # ================================
        # STORE COMPREHENSIVE METRICS
        # ================================
        epoch_metrics = {
            # Basic info
            'epoch': epoch,
            'time': epoch_time,
            
            # Loss metrics (maintaining backward compatibility)
            'train loss': train_total_loss,
            'val loss': val_total_loss,
            'val reconstruction loss': val_recon_loss,
            'val_kl_loss': val_kl_loss,
            'val classification loss': val_clf_loss,
            
            # Enhanced loss metrics
            'train_total_loss': train_total_loss,
            'train_reconstruction_loss': train_recon_loss,
            'train_kl_loss': train_kl_loss,
            'train_classification_loss': train_clf_loss,
            'val_total_loss': val_total_loss,
            'val_reconstruction_loss': val_recon_loss,
            'val_classification_loss': val_clf_loss,
            
            # NEW: Accuracy metrics
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_macro_f1': train_macro_f1,
            'val_macro_f1': val_macro_f1,
            'train_weighted_f1': train_weighted_f1,
            'val_weighted_f1': val_weighted_f1,
        }
        
        # Add per-class F1 scores if available
        epoch_metrics.update(per_class_f1)
        
        history.append(epoch_metrics)
        
        # ================================
        # ENHANCED LOGGING
        # ================================
        logging.info(f'Epoch {epoch+1}/{epochs} - '
                    f'Train Loss: {train_total_loss:.3f}, Val Loss: {val_total_loss:.3f} | '
                    f'Train Acc: {train_accuracy:.1f}%, Val Acc: {val_accuracy:.1f}% | '
                    f'ZINB: {val_recon_loss:.3f}, KL: {val_kl_loss:.3f}, '
                    f'Clf: {val_clf_loss:.3f} | Time: {epoch_time:.1f}s')
        
        # Log F1 scores every 5 epochs for detailed monitoring
        if (epoch + 1) % 5 == 0:
            logging.info(f'  └── F1 Scores - Train: {train_macro_f1:.1f}% (macro), '
                        f'Val: {val_macro_f1:.1f}% (macro)')
    
    return history


def calculate_final_metrics(model, val_dl, class_names=None):
    """
    Calculate comprehensive final metrics after training.
    Call this function after training completes.
    """
    model.eval()
    all_predictions = []
    all_true_labels = []
    use_cuda = next(model.parameters()).is_cuda

    with torch.no_grad():
        for batch in val_dl:
            x, _, _, y_true = batch['x'], batch['x_raw'], batch['size_factor'], batch['y']
            if use_cuda:
                x, y_true = x.cuda(), y_true.cuda()

            outputs = model(x)
            _, predicted = torch.max(outputs['y'], 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(y_true.cpu().numpy())

    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_true_labels, all_predictions) * 100
    macro_f1 = f1_score(all_true_labels, all_predictions, average='macro') * 100
    weighted_f1 = f1_score(all_true_labels, all_predictions, average='weighted') * 100

    # Detailed classification report
    class_report = classification_report(all_true_labels, all_predictions,
                                       target_names=class_names,
                                       output_dict=True, zero_division=0)

    print("\n FINAL CLASSIFICATION METRICS")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Macro F1-Score:   {macro_f1:.2f}%")
    print(f"Weighted F1-Score: {weighted_f1:.2f}%")

    if class_names:
        print(f"\n PER-CLASS PERFORMANCE:")
        for class_name in class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"  {class_name:15}: "
                     f"Precision={metrics['precision']*100:.1f}%, "
                     f"Recall={metrics['recall']*100:.1f}%, "
                     f"F1={metrics['f1-score']*100:.1f}%")

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'classification_report': class_report,
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }