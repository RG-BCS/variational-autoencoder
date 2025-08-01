def show_reconstructions(model, dataloader=None, sample_input=None, num_images=8):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if dataloader:
            batch = next(iter(dataloader))
            if isinstance(batch, (tuple, list)):  # if (x, y)
                batch = batch[0]
        else:
            batch = sample_input
        
        batch = batch[:num_images].to(device)
        recon, _, _ = model(batch)
        recon = torch.sigmoid(recon)
        comparison = torch.cat([batch.cpu(), recon.cpu()])
        grid = torchvision.utils.make_grid(comparison, nrow=num_images, normalize=False, pad_value=1)
        plt.figure(figsize=(20, 4))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title("Top: Original | Bottom: Reconstructed")
        plt.show()

def determine_norm_abnorm(input_img, recon_logits, threshold=None, loss_type='bce', return_scores=False):
    if loss_type == 'bce':
        recon_loss = F.binary_cross_entropy_with_logits(recon_logits, input_img, reduction='none')
        recon_loss = recon_loss.sum(dim=[1, 2, 3])  # per sample losses
    elif loss_type == 'mse':
        recon_logits = torch.sigmoid(recon_logits)
        recon_loss = F.mse_loss(recon_logits, input_img, reduction='none')
        recon_loss = recon_loss.sum(dim=[1, 2, 3])
    else:
        raise ValueError("loss_type must be 'bce' or 'mse'")

    if return_scores:
        return recon_loss.cpu().tolist()

    if threshold is None:
        raise ValueError("threshold must be provided when return_scores=False")

    preds = [1 if loss > threshold else 0 for loss in recon_loss]
    return preds

def compute_scores_and_labels(model, dataloader, loss_type='bce'):
    model.eval()
    device = next(model.parameters()).device
    y_true = []
    y_scores = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            recon_logits, _, _ = model(x)
            scores = determine_norm_abnorm(x, recon_logits, loss_type=loss_type, return_scores=True)
            y_scores.extend(scores)
            y_true.extend(y.cpu().tolist())

    return y_true, y_scores

def plot_confusion_matrix(model, dataloader, threshold, labels=['normal', 'abnormal'], 
                          normalize=False, title='Confusion Matrix', loss_type='bce'):
    model.eval()
    device = next(model.parameters()).device
    y_pred = []
    y_true = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            recon_logits, _, _ = model(x)
            preds = determine_norm_abnorm(x, recon_logits, threshold, loss_type)
            y_pred.extend(preds)
            y_true.extend(y.cpu().tolist())

    print(f"\n# y_true: {len(y_true)} | # y_pred: {len(y_pred)}\n")
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.2f' if normalize else 'd')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc_pr(model, dataloader, loss_type='bce'):
    
    y_true, y_scores = compute_scores_and_labels(model, dataloader, loss_type=loss_type)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

def get_recon_losses_per_image_after_training(model, dataloader, loss_type='bce'):
    """
    Computes per-image reconstruction loss from a trained VAE.
    Used to set anomaly detection threshold post-training.
    """
    if loss_type == 'bce':
        loss_fn = nn.BCELoss(reduction='none')
    elif loss_type == 'mse':
        loss_fn = nn.MSELoss(reduction='none')
    else:
        raise ValueError("loss_type must be 'bce' or 'mse'")

    model.eval()
    losses = []
    with torch.no_grad():
        for x in dataloader:
            if isinstance(x, (tuple, list)):
                x = x[0]
            x = x.to(device)
            recon_logits, _, _ = model(x)
            recon = torch.sigmoid(recon_logits)  # Always sigmoid for final output range [0,1]
            recon_loss = loss_fn(recon, x)  # shape: [B, C, H, W]
            per_image_loss = recon_loss.sum(dim=[1, 2, 3])  # sum across dimensions
            losses.extend(per_image_loss.cpu().numpy())
    return np.array(losses)

def compute_total_loss(input_img, recon_logits, mu, logvar, beta=1.0, free_bits=0.0, loss_type='bce',verbose=False):
    if loss_type == 'bce':
        # threshold = np.mean(train_loss) + 2*np.std(train_loss)
        recon_loss = F.binary_cross_entropy_with_logits(recon_logits, input_img, reduction='none')
        recon_loss = recon_loss.sum(dim=[1, 2, 3])  # shape: [batch_size]
        recon_loss = recon_loss.mean()
    elif loss_type == 'mse':
        recon_logits = torch.sigmoid(recon_logits)
        recon_loss = F.mse_loss(recon_logits, input_img, reduction='mean')
    else:
        raise ValueError("loss_type must be 'bce' or 'mse'")

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # KL divergence between posterior and N(0,1)
    if free_bits > 0:
        kl = torch.clamp(kl, min=free_bits)
    kl_loss = kl.sum(dim=1).mean()  # sum over latent dim, average over batch

    total_loss = recon_loss + beta * kl_loss

    if verbose:
        print(f"[Loss] total: {total_loss.item():.4f} | recon: {recon_loss.item():.4f}"\
              f" | KL: {kl_loss.item():.4f} | β: {beta:.2f}")
    return total_loss,recon_loss,beta * kl_loss

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_model(model, num_epochs, train_dl,val_dl,test_dl, optimizer,scheduler=None,clip_norm=False,
                max_norm = 1.0,loss_type='bce',max_beta=25.0,warmup_frac=0.5, free_bits=0.0,
                sample_images_for_interp=None):

    def cyclical_beta(epoch, cycle_length=20):
        return min(1.0, (epoch % cycle_length) / (cycle_length / 2))

    def linear_beta(epoch, total_epochs, max_beta=max_beta,warmup_frac=warmup_frac):
        ramp_up_epochs = int(total_epochs * warmup_frac)
        if epoch < ramp_up_epochs:
            return max_beta * (epoch / ramp_up_epochs)
        return max_beta

    model.train()
    total_train_loss,total_val_loss,train_recon_loss,train_kl_loss= [],[],[],[]
    start_time = time.time()
    for epoch in range(num_epochs):
        #beta = cyclical_beta(epoch)
        #beta = linear_beta(epoch, num_epochs)
        beta = 1e-4
        epoch_loss ,epoch_recon_loss,epoch_kl_loss = 0.,0.0,0.0
        for input_images,_ in train_dl:
            input_images = input_images.to(device)
            recon_img, mu, logvar = model(input_images)
            loss,recon_loss,kl_loss = compute_total_loss(input_images, recon_img, mu, logvar,beta,free_bits,loss_type)
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            norm_grad = grad_norm(model)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        epoch_loss /= len(train_dl.dataset)
        total_train_loss.append(epoch_loss)
        epoch_recon_loss /= len(train_dl.dataset)
        train_recon_loss.append(epoch_recon_loss)
        epoch_kl_loss /= len(train_dl.dataset)
        train_kl_loss.append(epoch_kl_loss)
        
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for val_batch,_ in val_dl:
                val_batch = val_batch.to(device)
                recon, mu, logvar = model(val_batch)
                loss,_,_ = compute_total_loss(val_batch, recon, mu, logvar,beta,free_bits,loss_type)
                val_loss += loss.item()
        val_loss /= len(val_dl.dataset)
        if scheduler:
            #scheduler.step(val_loss)
            scheduler.step()
        total_val_loss.append(val_loss)
        elapsed_time = (time.time()-start_time)/60
        if epoch % 50 == 0 or epoch ==num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}| Train_Loss: {epoch_loss:.4f}| Val_Loss:{val_loss:.4f}|"\
                  f" Grad_Norm: {norm_grad:.4f}| Time: {elapsed_time:.3f}min\n")
            show_reconstructions(model, test_dl,sample_input=None, num_images=20)
            start_time = time.time()
        model.train()
    return total_train_loss,total_val_loss,train_recon_loss,train_kl_loss
