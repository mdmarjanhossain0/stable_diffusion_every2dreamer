from pytorch_fid import fid_score

def main():
    path_to_real_images = './test_real'
    path_to_generated_images = './test_output'

    fid_value = fid_score.calculate_fid_given_paths(
        [path_to_real_images, path_to_generated_images],
        batch_size=1,
        device='cuda',
        dims=2048
    )

    print("FID:", fid_value)

if __name__ == "__main__":
    main()
