from config import Config
from gans.seqgan_instructor import SeqGANInstructor
from utils.text_process import text_process, load_test_dict


def main():
    opt = Config()

    if opt.if_real_data:
        opt.max_seq_len, opt.vocab_size = text_process('data/' + opt.dataset + '.txt')
        opt.extend_vocab_size = len(load_test_dict(opt.dataset)[0])  # init classifier vocab_size

    inst = SeqGANInstructor(opt)
    inst._run()


if __name__ == "__main__":

    for i in range(4600, 100, -400):
        main()
        with open("num_samples.txt", "w") as f:
            f.write(str(i))
