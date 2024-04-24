# %%
from flowerclient import FlowerClient
from sklearn.model_selection import train_test_split
from client import apply_smpc, data_preparation, decrypt_list_of_lists  # , encrypt

# %%
if __name__ == '__main__':
    # %%
    train_path = 'Airline Satisfaction/train.csv'
    test_path = 'Airline Satisfaction/test.csv'
    n_shares = 3
    k = 3
    m = 3
    dp = False
    type_ss = "shamir"  # "shamir" or "additif"
    # %%
    train_sets = data_preparation(train_path, 1)
    test_sets = data_preparation(test_path, 1)

    # %%
    x_train, y_train = train_sets[0]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,
                                                      stratify=y_train)
    x_test, y_test = test_sets[0]

    # %%
    flower_client = FlowerClient(256, x_train, x_val, x_test, y_train, y_val, y_test, dp, delta=1/(2 * len(x_train)))

    # %%
    old_params = flower_client.get_parameters({})
    res = old_params[:]
    for i in range(1):
        res = flower_client.fit(res, {})[0]
        loss = flower_client.evaluate(res, {})[0]

    # %%
    encrypted_result, dic_shapes = apply_smpc(res, n_shares, type_ss, k, m)

    # %%
    decrypted_result = decrypt_list_of_lists(encrypted_result, type_ss, dic_shapes, m)

    # %%
    print("Liste de listes originale :")
    print(res)

    # %%
    # print("\nListe de listes chiffrée :")
    # print(encrypted_result[0])
    # print(encrypted_result[1])
    # print(encrypted_result[2])

    print("\nListe de listes déchiffré :")
    print(decrypted_result)

    # %% verifier qu'on obtient bien le même résultat
    for layer, val in enumerate(res):
        print(f"Layer {layer} : {val.all() == decrypted_result[layer].all()}")
