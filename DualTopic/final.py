import os
import yaml
import scipy.io
import argparse
import numpy as np
# Sửa đường dẫn import nếu cần để phù hợp với cấu trúc chạy thực tế
from runners.Runner import Runner
from utils.data import file_utils
from utils.data.TextData import DatasetHandler

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Name of the model config file (e.g., DualTopic)')
    parser.add_argument('--dataset', required=True, help='Name of the dataset config file (e.g., Amazon_Review)')
    # Các tham số khác sẽ được load từ file config
    args = parser.parse_args()
    return args

def export_beta(beta, vocab, output_prefix, lang, num_top_word=15):
    """Exports topic words (beta) to a file."""
    topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=num_top_word)
    file_utils.save_text(topic_str_list, path=f'{output_prefix}_T{num_top_word}_{lang}.txt') # Thêm .txt
    return topic_str_list

def main():
    # --- 1. Parse Args & Load Configs ---
    args = parse_args()
    try:
        args = file_utils.update_args(args, f'./configs/model/{args.model}.yaml')
        args = file_utils.update_args(args, f'./configs/dataset/{args.dataset}.yaml')
    except FileNotFoundError as e:
        print(f"Error loading config file: {e}. Make sure paths are correct relative to execution directory.")
        return

    # --- 2. Validate Required Config Arguments ---
    # Các tham số bắt buộc phải có trong args sau khi load config
    required_params = [
        'num_topic', 'hidden_dim', 'dropout', 'tau', 'lambda_contrast',
        'gamma_align', 'epsilon', 'learning_rate', 'epochs', 'batch_size',
        'lang1', 'lang2'
    ]
    for param in required_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required parameter '{param}' in configuration files.")

    # Đặt giá trị mặc định cho tham số tùy chọn nếu thiếu
    args.align_start_epoch = getattr(args, 'align_start_epoch', args.epochs + 1) # Logic từ Runner
    args.lr_scheduler = getattr(args, 'lr_scheduler', None) # Scheduler là tùy chọn
    if args.lr_scheduler == 'StepLR':
        if not hasattr(args, 'lr_step_size') or not hasattr(args, 'lr_gamma'):
            raise ValueError("Missing 'lr_step_size' or 'lr_gamma' for StepLR scheduler.")

    # --- 3. Setup Output Path ---
    output_dir = f'output/{args.dataset}'
    # Đảm bảo tên file output rõ ràng và bao gồm các siêu tham số chính nếu cần
    output_prefix = f'{output_dir}/{args.model}_K{args.num_topic}_LR{args.learning_rate}_BS{args.batch_size}_LC{args.lambda_contrast}_GA{args.gamma_align}'
    file_utils.make_dir(output_dir)
    print("Running with configuration:")
    print(yaml.dump(vars(args), default_flow_style=False, indent=2))
    # Lưu config đã sử dụng vào thư mục output
    with open(f"{output_prefix}_config.yaml", 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


    # --- 4. Load Data (DatasetHandler, Vocab, Embeddings) ---
    data_root = f'./data/{args.dataset}'
    print(f"Loading data from: {data_root}")
    try:
        # Khởi tạo DatasetHandler (chỉ load BoW và cluster labels theo TextData.py)
        dataset_handler = DatasetHandler(args.dataset, args.batch_size, args.lang1, args.lang2)

        # Load vocabularies explicitly
        vocab_lang1 = file_utils.read_texts(os.path.join(data_root, f'vocab_{args.lang1}'))
        vocab_lang2 = file_utils.read_texts(os.path.join(data_root, f'vocab_{args.lang2}'))
        vocab_size_lang1 = len(vocab_lang1)
        vocab_size_lang2 = len(vocab_lang2)

        # Load pre-trained word embeddings explicitly
        word_embeddings_lang1 = np.load(os.path.join(data_root, f'word_embeddings_{args.lang1}.npy'))
        word_embeddings_lang2 = np.load(os.path.join(data_root, f'word_embeddings_{args.lang2}.npy'))
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Ensure all necessary data files (vocab, embeddings, bow) exist in {data_root}")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    print(f"Vocab size {args.lang1}: {vocab_size_lang1}, Vocab size {args.lang2}: {vocab_size_lang2}")
    print(f"Word embedding shape {args.lang1}: {word_embeddings_lang1.shape}")
    print(f"Word embedding shape {args.lang2}: {word_embeddings_lang2.shape}")
    print(f"Train data size: {dataset_handler.train_size}")
    print(f"Test data size: {len(dataset_handler.test_loader.dataset)}")

    # --- 5. Prepare Parameters for DualTopic Model ---
    # Phải khớp chính xác với thứ tự trong DualTopic.__init__
    model_params_list = [
        vocab_size_lang1,
        vocab_size_lang2,
        args.num_topic,
        args.hidden_dim,
        word_embeddings_lang1,
        word_embeddings_lang2,
        args.dropout,
        args.tau,
        args.lambda_contrast,
        args.gamma_align,
        args.epsilon
    ]

    # --- 6. Initialize Runner ---
    print("Initializing runner...")
    runner = Runner(args, model_params_list, args.lang1, args.lang2)

    # --- 7. Train Model ---
    print("Starting training...")
    result= runner.train(dataset_handler.train_loader)
    beta_lang1, beta_lang2 = result['beta_lang1'], result['beta_lang2']
    print("Training finished.")

    # --- 8. Export Beta (Topic Words) ---
    print("Exporting topic words (beta)...")
    topic_str_list_lang1 = export_beta(beta_lang1, vocab_lang1, output_prefix, args.lang1)
    topic_str_list_lang2 = export_beta(beta_lang2, vocab_lang2, output_prefix, args.lang2)

    # In ra một vài topic ví dụ
    print("\nSample Topics:")
    for i in range(min(5, args.num_topic)): # In tối đa 5 topic đầu
        print(f"--- Topic {i} ---")
        print(f"  {args.lang1.upper()}: {topic_str_list_lang1[i]}")
        print(f"  {args.lang2.upper()}: {topic_str_list_lang2[i]}")

    # --- 9. Evaluate (Get Theta for Train/Test) ---
    print("Evaluating on train and test sets (getting theta)...")
    # runner.test nhận DataLoader làm đầu vào
    train_theta_lang1, train_theta_lang2 = runner.test(dataset_handler.train_loader)
    test_theta_lang1, test_theta_lang2 = runner.test(dataset_handler.test_loader)
    print("Evaluation finished.")

    # --- 10. Save Results ---
    print("Saving results...")
    rst_dict = {
        f'beta_{args.lang1}': beta_lang1,
        f'beta_{args.lang2}': beta_lang2,
        f'train_theta_{args.lang1}': train_theta_lang1,
        f'train_theta_{args.lang2}': train_theta_lang2,
        f'test_theta_{args.lang1}': test_theta_lang1,
        f'test_theta_{args.lang2}': test_theta_lang2,
        # Có thể thêm cả vocab vào file mat nếu cần
        # f'vocab_{args.lang1}': vocab_lang1,
        # f'vocab_{args.lang2}': vocab_lang2,
    }
    save_path = f'{output_prefix}_results.mat' # Đổi tên file kết quả
    try:
        scipy.io.savemat(save_path, rst_dict, do_compression=True) # Thêm nén để tiết kiệm dung lượng
        print(f"Results successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving results to .mat file: {e}")

if __name__ == '__main__':
    main()