from torch_fidelity import calculate_metrics

if __name__ == "__main__":
    metrics = calculate_metrics(
        input1="./test_real",
        input2="./test_output",
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        pr=True,
        pr_subset_size=5,    # important for small dataset
        kid_subset_size=5
    )


    # Safely print metrics
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
