def extract_results(last_line):
    parts = last_line.split(',')
    # Mean-Last-Epoch-Val e MEE-Val
    mean_last_epoch_val = float(parts[1].split(': ')[1])
    mee_val = float(parts[3].split(': ')[1])
    return mean_last_epoch_val, mee_val

def calculate_score(mean_last_epoch_val, mee_val):
    return (mean_last_epoch_val + mee_val) / 2

def read_and_analyze(file_name):
    with open(file_name, 'r') as file:
        content = file.read()

    # Split test
    tests = content.split('\n[')[1:]
    test_results = []

    for test in tests:
        lines = test.strip().split('\n')
        config = '[' + lines[0]
        result_line = lines[-1]
        mean_last_epoch_val, mee_val = extract_results(result_line)
        score = calculate_score(mean_last_epoch_val, mee_val)
        test_results.append((config, score))

    # Sorting by score
    sorted_tests = sorted(test_results, key=lambda x: x[1])

    # Take only 10 results
    return sorted_tests[:10]

best_tests = read_and_analyze("TEST15/Summary.txt")
for config, score in best_tests:
    print(f"Configurazione: {config} - Score: {score}")
