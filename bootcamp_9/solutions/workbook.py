import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import random
    import os
    from pathlib import Path

    return (
        DataLoader,
        MinMaxScaler,
        Path,
        TensorDataset,
        mean_absolute_percentage_error,
        mean_squared_error,
        mo,
        nn,
        np,
        os,
        pd,
        plt,
        random,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # PJM HOURLY ENERGY CONSUMPTION

    https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data
    """)
    return


@app.cell
def _(Path, mo, pd):
    energy_data = pd.read_csv(Path(mo.notebook_location()).resolve().parent / 'AEP_hourly.csv')
    energy_data = energy_data.rename(columns={'Datetime': 'date', 'AEP_MW': 'mw'})
    energy_data['date'] = pd.to_datetime(energy_data['date'])
    energy_data = energy_data.sort_values('date')

    print(energy_data.head())
    return (energy_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## VISUALIZATION
    Let's plot some graphs and see what we are working with
    """)
    return


@app.cell
def _(energy_data, plt):
    visual_df = energy_data.copy()

    # plot the entire dataset
    plt.figure(figsize=(15, 5))
    plt.plot(visual_df['date'], visual_df['mw'])
    plt.title("Energy Consumption over ~10 Years")
    plt.xlabel("Date")
    plt.ylabel("MW Load")
    plt.show()


    # plot one year
    visual_zoom_subset = visual_df[(visual_df['date'] > '2016-01-01') & (visual_df['date'] < '2017-01-01')]

    plt.figure(figsize=(15, 5))
    plt.plot(visual_zoom_subset['date'], visual_zoom_subset['mw'], color='orange')
    plt.title("Zoomed View: One Year of Data")
    plt.xlabel("Date")
    plt.ylabel("MW Load")
    plt.grid(True)
    plt.show()

    # plot one month
    visual_zoom_subset = visual_df[(visual_df['date'] > '2016-01-01') & (visual_df['date'] < '2016-02-01')]

    plt.figure(figsize=(15, 5))
    plt.plot(visual_zoom_subset['date'], visual_zoom_subset['mw'], color='orange')
    plt.title("Zoomed View: One Month of Data")
    plt.xlabel("Date")
    plt.ylabel("MW Load")
    plt.grid(True)
    plt.show()

    # plot one week
    visual_zoom_subset = visual_df[(visual_df['date'] > '2016-01-01') & (visual_df['date'] < '2016-01-08')]

    plt.figure(figsize=(15, 5))
    plt.plot(visual_zoom_subset['date'], visual_zoom_subset['mw'], color='orange')
    plt.title("Zoomed View: One Week of Data")
    plt.xlabel("Date")
    plt.ylabel("MW Load")
    plt.grid(True)
    plt.show()


    # plot one day
    visual_zoom_subset = visual_df[(visual_df['date'] > '2016-01-01') & (visual_df['date'] < '2016-01-02')]

    plt.figure(figsize=(15, 5))
    plt.plot(visual_zoom_subset['date'], visual_zoom_subset['mw'], color='orange')
    plt.title("Zoomed View: One Day of Data")
    plt.xlabel("Date")
    plt.ylabel("MW Load")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(energy_data, pd, plt):
    visual_overlap_df = energy_data.copy()



    # Ensure the date column is in datetime format (if not already)
    visual_overlap_df['date'] = pd.to_datetime(visual_overlap_df['date'])

    # Pick a starting point for your overlaps (change this to whatever date you prefer)
    visual_start_date = pd.to_datetime('2016-01-01')

    # -------------------------------------------------------------------
    # Plot 1: 10 Consecutive Days Overlapped
    # -------------------------------------------------------------------
    plt.figure(figsize=(15, 6))
    for visual_i in range(10):
        visual_day_start = visual_start_date + pd.Timedelta(days=visual_i)
        visual_day_end = visual_day_start + pd.Timedelta(days=1)

        visual_overlap_subset = visual_overlap_df[(visual_overlap_df['date'] >= visual_day_start) & (visual_overlap_df['date'] < visual_day_end)].copy()
        if not visual_overlap_subset.empty:
            visual_overlap_subset['hour_of_day'] = visual_overlap_subset['date'].dt.hour + (visual_overlap_subset['date'].dt.minute / 60.0)
            plt.plot(visual_overlap_subset['hour_of_day'], visual_overlap_subset['mw'], label=visual_day_start.strftime('%a, %b %d'))

    plt.title("Seasonality: 10 Consecutive Days Overlapped")
    plt.xlabel("Hour of Day (0 - 23)")
    plt.ylabel("MW Load")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------
    # Plot 2: 10 Consecutive Weeks Overlapped (Daily Averages)
    # -------------------------------------------------------------------
    plt.figure(figsize=(15, 6))
    for visual_i in range(10):
        visual_week_start = visual_start_date + pd.Timedelta(weeks=visual_i)
        visual_week_end = visual_week_start + pd.Timedelta(weeks=1)

        visual_overlap_subset = visual_overlap_df[(visual_overlap_df['date'] >= visual_week_start) & (visual_overlap_df['date'] < visual_week_end)].copy()
        if not visual_overlap_subset.empty:
            # Group by the day of the week (0=Monday, 6=Sunday) and calculate the mean
            visual_overlap_subset['day_of_week'] = visual_overlap_subset['date'].dt.dayofweek
            visual_daily_avg = visual_overlap_subset.groupby('day_of_week')['mw'].mean()

            plt.plot(
                visual_daily_avg.index,
                visual_daily_avg.values,
                marker='o',
                label=f"Week of {visual_week_start.strftime('%b %d')}"
            )

    plt.title("Seasonality: 10 Consecutive Weeks Overlapped (Daily Averages)")
    plt.xlabel("Day of the Week (0 = Monday, 6 = Sunday)")
    plt.ylabel("Average MW Load")
    # Explicitly set x-ticks so it shows whole numbers for days
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------
    # Plot 3: 10 Consecutive Months Overlapped (Weekly Averages)
    # -------------------------------------------------------------------
    plt.figure(figsize=(15, 6))
    for visual_i in range(10):
        visual_month_start = visual_start_date + pd.DateOffset(months=visual_i)
        visual_month_end = visual_start_date + pd.DateOffset(months=visual_i+1)

        visual_overlap_subset = visual_overlap_df[(visual_overlap_df['date'] >= visual_month_start) & (visual_overlap_df['date'] < visual_month_end)].copy()
        if not visual_overlap_subset.empty:
            # Calculate the week of the month (1, 2, 3, 4, 5)
            # Days 1-7 -> Week 1; Days 8-14 -> Week 2, etc.
            visual_overlap_subset['week_of_month'] = ((visual_overlap_subset['date'].dt.day - 1) // 7) + 1
            visual_weekly_avg = visual_overlap_subset.groupby('week_of_month')['mw'].mean()

            plt.plot(
                visual_weekly_avg.index,
                visual_weekly_avg.values,
                marker='o',
                label=visual_month_start.strftime('%B %Y')
            )

    plt.title("Seasonality: 10 Consecutive Months Overlapped (Weekly Averages)")
    plt.xlabel("Week of the Month (1 - 5)")
    plt.ylabel("Average MW Load")
    plt.xticks(range(1, 6)) # Ensures x-axis only shows 1, 2, 3, 4, 5
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()





    # Ensure the date column is in datetime format (if not already)
    # df['date'] = pd.to_datetime(df['date'])

    visual_start_date = pd.to_datetime('2006-01-01')

    plt.figure(figsize=(15, 6))

    for visual_i in range(10):
        visual_year_start = visual_start_date + pd.DateOffset(years=visual_i)
        visual_year_end = visual_start_date + pd.DateOffset(years=visual_i+1)

        # Get the data for the specific year
        visual_overlap_subset = visual_overlap_df[(visual_overlap_df['date'] >= visual_year_start) & (visual_overlap_df['date'] < visual_year_end)].copy()

        if not visual_overlap_subset.empty:
            # Group by the month (1-12) and calculate the average MW load
            visual_monthly_avg = visual_overlap_subset.groupby(visual_overlap_subset['date'].dt.month)['mw'].mean()

            # Plot the monthly averages
            # monthly_avg.index will be the month numbers (1 to 12)
            # monthly_avg.values will be the average MW loads
            plt.plot(visual_monthly_avg.index, visual_monthly_avg.values, label=visual_year_start.strftime('%Y'), marker='o')

    plt.title("Seasonality: Monthly Average MW Load (10 Years Overlapped)")
    plt.xlabel("Month of the Year")
    plt.ylabel("Average MW Load")

    # Ensure x-axis only shows integer months from 1 to 12
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## SIMPLE NN
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Prep the data first: we are going to use two years of data rather than the whole 10, to make it faster.

    We are normalizing to with a MinMaxScaler for better handling of data by the model

    We are also creating a function that creates the sliding window data we need as input for the model
    """)
    return


@app.cell
def _(MinMaxScaler, energy_data, np):
    simple_df = energy_data.copy()

    simple_df_small = simple_df[simple_df['date'] < '2017-01-01']
    simple_df_small = simple_df_small[simple_df_small['date'] > '2015-01-01']

    simple_train_size = int(len(simple_df_small) * 0.8)
    simple_train = simple_df_small[['mw']].iloc[:simple_train_size].values
    simple_test = simple_df_small[['mw']].iloc[simple_train_size:].values

    # remember to fit the scaler only on the training data to avoid data leakage
    simple_scaler = MinMaxScaler()
    simple_scaler.fit(simple_train)

    simple_train_normalized = simple_scaler.transform(simple_train)
    simple_test_normalized = simple_scaler.transform(simple_test)

    def simple_create_sequences(data, window_size):
        inputs = []
        targets = []
        for simple_i in range(len(data) - window_size):
            x = data[simple_i : simple_i + window_size]
            y = data[simple_i + window_size]
            inputs.append(x)
            targets.append(y)
        return np.array(inputs), np.array(targets)

    return (
        simple_create_sequences,
        simple_scaler,
        simple_test_normalized,
        simple_train_normalized,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Simple NN on 24 hours of data
    """)
    return


@app.cell
def _(
    simple_create_sequences,
    simple_test_normalized,
    simple_train_normalized,
    torch,
):
    simple_WINDOW_SIZE = 24
    simple_X_train, simple_y_train = simple_create_sequences(simple_train_normalized, simple_WINDOW_SIZE)
    simple_X_test, simple_y_test = simple_create_sequences(simple_test_normalized, simple_WINDOW_SIZE)

    simple_X_train_tensor = torch.FloatTensor(simple_X_train)
    simple_y_train_tensor = torch.FloatTensor(simple_y_train)
    simple_X_test_tensor = torch.FloatTensor(simple_X_test)
    simple_y_test_tensor = torch.FloatTensor(simple_y_test)
    return (
        simple_WINDOW_SIZE,
        simple_X_test_tensor,
        simple_X_train_tensor,
        simple_y_test_tensor,
        simple_y_train_tensor,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    architecture
    """)
    return


@app.cell
def _(nn):
    class SimpleMLP(nn.Module):
        # nn.Module is the base class for all neural networks in PyTorch
        # we need an __init__ and a forward method
        def __init__(self, input_size):
            # super is a function to turn our custom model into a child of the nn.Module parent class
            super(SimpleMLP, self).__init__()
            # here we define the layers of our network
            self.layer1 = nn.Linear(input_size, 128)
            # BRAND RECOGNITION
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(128, 1)

        def forward(self, x):
            # .view is a flattener, it takes the input and reshapes it to be (batch_size, -1)
            # where -1 means flatten everything else (but the batch size)
            x = x.view(x.shape[0], -1)
            # here we put the layers in order (before we only defined them)
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    return (SimpleMLP,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Training
    we are going to use MSE for the loss, and the Adam optimizer
    """)
    return


@app.cell
def _(
    SimpleMLP,
    nn,
    simple_WINDOW_SIZE,
    simple_X_train_tensor,
    simple_y_train_tensor,
    torch,
):
    simple_model = SimpleMLP(input_size=simple_WINDOW_SIZE)
    # we use MLE as a Loss function
    simple_criterion = nn.MSELoss()
    # we use the AdamW optimizer because google says so (it's an improvement over Adam)
    simple_optimizer = torch.optim.AdamW(simple_model.parameters(), lr=0.001)

    simple_epochs = 50
    print("Training...")
    for simple_epoch in range(simple_epochs):
        # reset the gradients to zero before each step
        simple_optimizer.zero_grad()
        # output of the model (predictions)
        simple_output = simple_model(simple_X_train_tensor)
        # calculate loss
        simple_loss = simple_criterion(simple_output, simple_y_train_tensor)
        # backpropagate the loss to compute gradients
        simple_loss.backward()
        # optimization step (update the weights)
        simple_optimizer.step()
        if simple_epoch % 10 == 0:
            print(f"Epoch {simple_epoch} Loss: {simple_loss.item():.5f}")
    return (simple_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Testing
    We need to remember to convert back the results into MegaWatts

    We also plot the first and last hours of the test to see how the model did
    """)
    return


@app.cell
def _(
    np,
    plt,
    simple_X_test_tensor,
    simple_model,
    simple_scaler,
    simple_y_test_tensor,
    torch,
):
    simple_model.eval()

    simple_current_sequence = simple_X_test_tensor[0].unsqueeze(0)
    simple_predictions = []

    with torch.no_grad():
        for simple_i in range(len(simple_y_test_tensor)):
            simple_next_pred = simple_model(simple_current_sequence)
            simple_predictions.append(simple_next_pred.item())

            if len(simple_current_sequence.shape) == 3 and len(simple_next_pred.shape) == 2:
                simple_next_pred_reshaped = simple_next_pred.unsqueeze(1)
            else:
                simple_next_pred_reshaped = simple_next_pred

            simple_current_sequence = torch.cat((simple_current_sequence[:, 1:], simple_next_pred_reshaped), dim=1)

    simple_test_predictions_mw = simple_scaler.inverse_transform(np.array(simple_predictions).reshape(-1, 1))
    simple_y_test_mw = simple_scaler.inverse_transform(simple_y_test_tensor.numpy())


    # plot the first 24 hours
    plt.figure(figsize=(15, 6))
    plt.plot(simple_y_test_mw[:24], label='Actual Energy Usage', color='blue', linewidth=2)
    plt.plot(simple_test_predictions_mw[:24], label='Recursive Prediction', color='red', linestyle='--')
    plt.title("Recursive Eval: Prediction vs Reality (First 24 Hours)")
    plt.xlabel("Hours")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.show()


    # plot the first week
    plt.figure(figsize=(15, 6))
    plt.plot(simple_y_test_mw[:168], label='Actual Energy Usage', color='blue', linewidth=2)
    plt.plot(simple_test_predictions_mw[:168], label='Recursive Prediction', color='red', linestyle='--')
    plt.title("Recursive Eval: Prediction vs Reality (First Week)")
    plt.xlabel("Hours")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot the first month
    plt.figure(figsize=(15, 6))
    plt.plot(simple_y_test_mw[:720], label='Actual Energy Usage', color='blue', linewidth=2)
    plt.plot(simple_test_predictions_mw[:720], label='Recursive Prediction', color='red', linestyle='--')
    plt.title("Recursive Eval: Prediction vs Reality (First Month)")
    plt.xlabel("Hours")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot the last 200 hours
    plt.figure(figsize=(15, 6))
    plt.plot(simple_y_test_mw[-200:], label='Actual Energy Usage', color='blue', linewidth=2)
    plt.plot(simple_test_predictions_mw[-200:], label='Recursive Prediction', color='red', linestyle='--')
    plt.title("Recursive Eval: Prediction vs Reality (Last 200 Hours)")
    plt.xlabel("Hours")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # RNN

    we now try to work with a different model, the recurrent neural network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### RNN on 24 hours of data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Add two sinusoidal features for daily seasonality
    """)
    return


@app.cell
def _(np, os, random, torch):
    def rnn_set_seed(seed_value=42):
        """Set seed for reproducibility."""
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

    rnn_set_seed(42)
    return


@app.cell
def _(MinMaxScaler, energy_data, np):
    rnn_df_small = energy_data[(energy_data['date'] > '2015-01-01') & (energy_data['date'] < '2017-01-01')].copy()

    rnn_df_small['hour_sin'] = np.sin(2 * np.pi * rnn_df_small['date'].dt.hour / 24.0)
    rnn_df_small['hour_cos'] = np.cos(2 * np.pi * rnn_df_small['date'].dt.hour / 24.0)

    rnn_train_size = int(len(rnn_df_small) * 0.8)
    rnn_train_df = rnn_df_small.iloc[:rnn_train_size]
    rnn_test_df = rnn_df_small.iloc[rnn_train_size:]

    rnn_scaler = MinMaxScaler()
    rnn_train_mw_scaled = rnn_scaler.fit_transform(rnn_train_df[['mw']].values)
    rnn_test_mw_scaled = rnn_scaler.transform(rnn_test_df[['mw']].values)

    rnn_train_multivariate = np.hstack((rnn_train_mw_scaled, rnn_train_df[['hour_sin', 'hour_cos']].values))
    rnn_test_multivariate = np.hstack((rnn_test_mw_scaled, rnn_test_df[['hour_sin', 'hour_cos']].values))

    # we modify one line from before to account for new features
    def rnn_create_sequences(data, window_size):
        inputs = []
        targets = []
        for rnn_i in range(len(data) - window_size):
            x = data[rnn_i : rnn_i + window_size]
            y = data[rnn_i + window_size, 0]
            inputs.append(x)
            targets.append(y)
        return np.array(inputs), np.array(targets)

    return (
        rnn_create_sequences,
        rnn_scaler,
        rnn_test_multivariate,
        rnn_train_multivariate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Architecture
    """)
    return


@app.cell
def _(nn, torch):
    class VanillaRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            # Pytorch syntax for a NN model
            super(VanillaRNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            # batch_first=True means the input shape should be (batch_size, seq_length, input_size)
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            # layer for the final output
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # creates an initial hidden state (memory) made of zeros for the first iteration
            # to(x.device) ensures it's on the same device as the input (CPU or GPU)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            # out, _ because rnn returns two things and we don't need the second one
            out, _ = self.rnn(x, h0)
            # grab all batches, but then only the last time step
            # the RNN goes through every timestep from 1 to 24 but we only need the last one for the prediction
            # it then grabs all features
            out = self.fc(out[:, -1, :])
            return out

    return (VanillaRNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Training
    """)
    return


@app.cell
def _(
    VanillaRNN,
    nn,
    rnn_create_sequences,
    rnn_test_multivariate,
    rnn_train_multivariate,
    torch,
):
    rnn_WINDOW_SIZE = 24
    rnn_X_train, rnn_y_train = rnn_create_sequences(rnn_train_multivariate, rnn_WINDOW_SIZE)
    rnn_X_test, rnn_y_test = rnn_create_sequences(rnn_test_multivariate, rnn_WINDOW_SIZE)

    # we need the .view(-1, 1) to make sure the target is the right shape for MSELoss (which expects [batch_size, output_size])
    # instead of just  [batch_size]
    rnn_X_train_tensor = torch.FloatTensor(rnn_X_train)
    rnn_y_train_tensor = torch.FloatTensor(rnn_y_train).view(-1, 1)
    rnn_X_test_tensor = torch.FloatTensor(rnn_X_test)
    rnn_y_test_tensor = torch.FloatTensor(rnn_y_test).view(-1, 1)

    rnn_model = VanillaRNN(input_size=3, hidden_size=128, output_size=1)
    rnn_criterion = nn.MSELoss()
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)

    rnn_epochs = 50
    print("Training Multivariate RNN...")
    for rnn_epoch in range(rnn_epochs):
        rnn_model.train()
        rnn_optimizer.zero_grad()
        rnn_output = rnn_model(rnn_X_train_tensor)
        rnn_loss = rnn_criterion(rnn_output, rnn_y_train_tensor)
        rnn_loss.backward()
        rnn_optimizer.step()

        if rnn_epoch % 10 == 0:
            print(f"Epoch {rnn_epoch}, Loss: {rnn_loss.item():.5f}")
    return rnn_X_test_tensor, rnn_model, rnn_y_test_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Prediction
    """)
    return


@app.cell
def _(np, rnn_X_test_tensor, rnn_model, rnn_scaler, rnn_y_test_tensor, torch):
    rnn_model.eval()
    rnn_sequential_predictions = []

    # we copy (clone) the first test sequence of [24, 3], but we turn it into [1, 24, 3] so that Pytorch can process it
    rnn_current_seq = rnn_X_test_tensor[0].unsqueeze(0).clone()

    print("Generating sequential predictions...")
    # torch.no_grad() makes it so Pytorch doesn't calculate gradients in this block (don't need to, it's prediction)
    with torch.no_grad():
        for rnn_i in range(len(rnn_y_test_tensor)):
            rnn_pred_mw = rnn_model(rnn_current_seq)
            # .item() takes the single numerical value from the tensor, and we append it
            rnn_sequential_predictions.append(rnn_pred_mw.item())
            if rnn_i + 1 < len(rnn_X_test_tensor):
                # here we take the sinusoidal features from the test set so we don't have to recalculate them
                # takes the next 24 hours, takes the last hour and all 3 features. In addition, it turns it into [1, 1, 3]
                # so that Pytorch can process it
                rnn_next_features = rnn_X_test_tensor[rnn_i + 1, -1, :].clone().unsqueeze(0).unsqueeze(0)
                # insert into it the new prediction value we just got
                rnn_next_features[0, 0, 0] = rnn_pred_mw.item()
                # now we concatenate the sequence with the new features
                # we end up creating a sequence that contains our predictions rather than the actual values
                rnn_current_seq = torch.cat((rnn_current_seq[:, 1:, :], rnn_next_features), dim=1)

    # equivalent of .view(-1, 1) to make sure it's the right shape
    rnn_sequential_predictions = np.array(rnn_sequential_predictions).reshape(-1, 1)
    # inverse the MinMaxScaler transformation
    rnn_test_predictions_mw = rnn_scaler.inverse_transform(rnn_sequential_predictions)
    rnn_y_test_mw = rnn_scaler.inverse_transform(rnn_y_test_tensor.numpy())
    return rnn_test_predictions_mw, rnn_y_test_mw


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Plotting and evaluation measures
    """)
    return


@app.cell
def _(plt, rnn_test_predictions_mw, rnn_y_test_mw):
    plt.figure(figsize=(15, 6))
    plt.plot(rnn_y_test_mw[:300], label='Actual MW', color='blue', alpha=0.7)
    plt.plot(rnn_test_predictions_mw[:300], label='Sequential RNN Prediction', color='red', linestyle='--')
    plt.title("Vanilla RNN Sequential Prediction (Anchored with Time Features)")
    plt.xlabel("Hours into Test Set")
    plt.ylabel("MW Load")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(15, 6))
    plt.plot(rnn_y_test_mw, label='Actual MW', color='blue', alpha=0.7)
    plt.plot(rnn_test_predictions_mw, label='Sequential RNN Prediction', color='red', linestyle='--')
    plt.title("Vanilla RNN Sequential Prediction (Anchored with Time Features)")
    plt.xlabel("Hours into Test Set")
    plt.ylabel("MW Load")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We still need a concrete evaluation of the model, and for that we use MSE and MAPE
    We want to beat some baseline "dumb" models to make sure we got something good
    """)
    return


@app.cell
def _(mean_squared_error, np, rnn_test_predictions_mw, rnn_y_test_mw):
    rnn_mse = mean_squared_error(rnn_y_test_mw, rnn_test_predictions_mw)
    rnn_rmse = np.sqrt(rnn_mse)
    rnn_mape = np.mean(np.abs((rnn_y_test_mw - rnn_test_predictions_mw) / rnn_y_test_mw)) * 100

    print(f"--- Model Performance ---")
    print(f"RMSE: {rnn_rmse:.2f} MW")
    print(f"MAPE: {rnn_mape:.2f}%")

    # np.roll shifts the array by 1 (we take the previous hour's value)
    rnn_y_naive = np.roll(rnn_y_test_mw, 1)
    # The first element is now to be ignored
    rnn_baseline_rmse = np.sqrt(mean_squared_error(rnn_y_test_mw[1:], rnn_y_naive[1:]))

    print(f"\n--- Baseline Comparison ---")
    print(f"Baseline (Persistence) RMSE: {rnn_baseline_rmse:.2f} MW")

    if rnn_rmse < rnn_baseline_rmse:
        print(f"SUCCESS: Model beat the baseline by {rnn_baseline_rmse - rnn_rmse:.2f} MW")
    else:
        print(f"FAILURE: Model is worse than just guessing 'same as last hour'.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # LSTM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    we now try to use the LSTM model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We add:
    - clock features with sine and cosine
    - 24 hour chunk prediction
    - lag features of some hours back, one week, one year and two years back
    """)
    return


@app.cell
def _(np, os, random, torch):
    def lstm_set_seed(seed_value=42):
        """Set seed for reproducibility."""
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

    lstm_set_seed(42)
    return


@app.cell
def _(MinMaxScaler, energy_data, np):
    lstm_df_small = energy_data[(energy_data['date'] < '2017-01-01') &
                           (energy_data['date'] > '2007-01-01')].copy()

    lstm_day_year = lstm_df_small['date'].dt.dayofyear
    lstm_df_small['sin_year'] = np.sin(2 * np.pi * lstm_day_year / 365)
    lstm_df_small['cos_year'] = np.cos(2 * np.pi * lstm_day_year / 365)
    lstm_df_small['sin_hour'] = np.sin(2 * np.pi * lstm_df_small['date'].dt.hour / 24)
    lstm_df_small['cos_hour'] = np.cos(2 * np.pi * lstm_df_small['date'].dt.hour / 24)
    lstm_df_small['sin_day'] = np.sin(2 * np.pi * lstm_df_small['date'].dt.dayofweek / 7)
    lstm_df_small['cos_day'] = np.cos(2 * np.pi * lstm_df_small['date'].dt.dayofweek / 7)

    lstm_feature_cols = ['mw', 'sin_year', 'cos_year', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day']

    lstm_train_size = int(len(lstm_df_small) * 0.8)
    # .values returns a pure numpy array
    lstm_train_raw = lstm_df_small[lstm_feature_cols].iloc[:lstm_train_size].values
    lstm_test_raw = lstm_df_small[lstm_feature_cols].iloc[lstm_train_size:].values

    lstm_scaler = MinMaxScaler()
    # .reshape(-1, 1) turns 1D arrays into 2D arrays with one column (depends on what a function expects)
    lstm_scaler.fit(lstm_train_raw[:, 0].reshape(-1, 1))

    lstm_train_normalized = lstm_train_raw.copy()
    lstm_test_normalized = lstm_test_raw.copy()

    # .flatten() returns the 2D array back into a 1D array after scaling
    lstm_train_normalized[:, 0] = lstm_scaler.transform(lstm_train_raw[:, 0].reshape(-1, 1)).flatten()
    lstm_test_normalized[:, 0] = lstm_scaler.transform(lstm_test_raw[:, 0].reshape(-1, 1)).flatten()
    return (
        lstm_df_small,
        lstm_scaler,
        lstm_test_normalized,
        lstm_train_normalized,
        lstm_train_size,
    )


@app.cell
def _(lstm_train_normalized, np, torch):
    def lstm_create_chunk_sequences(data):
        inputs = []
        targets = []
        max_lookback = 24 * 365 * 2
        prediction_window = 24
        for i in range(max_lookback, len(data) - prediction_window):
            lstm_indices = []
            # these are our lag features
            for lstm_h in range(3, 0, -1): lstm_indices.append(i - lstm_h)
            lstm_indices.append(i - 24)
            lstm_indices.append(i - (24 * 7))
            lstm_indices.append(i - (24 * 365))
            lstm_indices.append(i - (24 * 365 * 2))

            inputs.append(data[lstm_indices])
            targets.append(data[i : i + prediction_window, 0])

        return np.array(inputs), np.array(targets)

    lstm_X_train, lstm_y_train = lstm_create_chunk_sequences(lstm_train_normalized)

    lstm_X_train_tensor = torch.FloatTensor(lstm_X_train)
    lstm_y_train_tensor = torch.FloatTensor(lstm_y_train)
    return (
        lstm_X_train_tensor,
        lstm_create_chunk_sequences,
        lstm_y_train_tensor,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Architecture
    """)
    return


@app.cell
def _(nn, torch):
    class LSTMModel(nn.Module):
        # notice input_size=7 because we have 7 features (mw + 6 sinusoidal features)
        def __init__(self, input_size=7, hidden_size=128, output_size=24):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=6, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # like the RNN, we need to create initial hidden and cell states for the LSTM
            h0 = torch.zeros(6, x.size(0), 128).to(x.device)
            c0 = torch.zeros(6, x.size(0), 128).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    return (LSTMModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Training
    """)
    return


@app.cell
def _(
    DataLoader,
    LSTMModel,
    TensorDataset,
    lstm_X_train_tensor,
    lstm_y_train_tensor,
    nn,
    torch,
):
    lstm_model = LSTMModel()
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_criterion = nn.MSELoss()

    print("Training...")
    lstm_train_dataset = TensorDataset(lstm_X_train_tensor, lstm_y_train_tensor)
    lstm_train_loader = DataLoader(lstm_train_dataset, batch_size=64, shuffle=True)

    for lstm_epoch in range(5):
        lstm_model.train()
        for lstm_batch_X, lstm_batch_y in lstm_train_loader:
            lstm_optimizer.zero_grad()
            lstm_loss = lstm_criterion(lstm_model(lstm_batch_X), lstm_batch_y)
            lstm_loss.backward()
            lstm_optimizer.step()
        print(f"Epoch {lstm_epoch+1} Loss: {lstm_loss.item():.5f}")
    return (lstm_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Prediction
    """)
    return


@app.cell
def _(lstm_model, lstm_test_normalized, lstm_train_normalized, np, torch):
    lstm_model.eval()

    # turns the training data into a list of rows
    lstm_history = [lstm_row for lstm_row in lstm_train_normalized]
    lstm_predictions = []
    # this buffer is so we can extract the sinusoidal features more easily
    lstm_lookback_buffer = lstm_train_normalized[-(24*365*2):]
    # attaches the 2 years of lookback to the beginning of the test so we can extract the lag features
    lstm_test_with_history = np.concatenate((lstm_lookback_buffer, lstm_test_normalized))
    # calcs exactly how many times we need to run the loop
    lstm_total_test_hours = len(lstm_test_normalized)
    lstm_num_chunks = int(np.ceil(lstm_total_test_hours / 24))
    # takes only the sinusoidal features
    lstm_future_time_features = lstm_test_normalized[:, 1:]

    print(f"Predicting {lstm_num_chunks} chunks (Total ~{lstm_num_chunks*24} hours)...")

    with torch.no_grad():
        for lstm_chunk_idx in range(lstm_num_chunks):
            lstm_t = len(lstm_history)
            lstm_indices = []
            # lag features
            for lstm_h in range(3, 0, -1): lstm_indices.append(lstm_t - lstm_h)
            lstm_indices.append(lstm_t - 24)
            lstm_indices.append(lstm_t - (24 * 7))
            lstm_indices.append(lstm_t - (24 * 365))
            lstm_indices.append(lstm_t - (24 * 365 * 2))

            # takes the indexes from history (lag features)
            lstm_seq_data = np.array([lstm_history[lstm_idx] for lstm_idx in lstm_indices])
            # turns them into a tensor and adds a dimension so it's [1, 7] instead of [7]
            lstm_current_input = torch.FloatTensor(lstm_seq_data).unsqueeze(0)
            # gets the 24 hour predicitons
            lstm_chunk_pred = lstm_model(lstm_current_input).numpy()[0]
            # add the predictions to the list of all predictions
            lstm_predictions.extend(lstm_chunk_pred)
            # moevs to the next 24 hours
            lstm_start_t = lstm_chunk_idx * 24
            lstm_end_t = lstm_start_t + 24
            lstm_current_chunk_times = lstm_future_time_features[lstm_start_t : lstm_end_t]

            # check that it doesn't go beyond if it doesn't have 24 hours left available
            if len(lstm_current_chunk_times) < 24:
                break

            # stack teh predictions with the sinusoidal features so we can add them to history
            lstm_new_rows = np.column_stack((lstm_chunk_pred, lstm_current_chunk_times))

            for lstm_row in lstm_new_rows:
                lstm_history.append(lstm_row)
    return (lstm_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Plotting and Evaluation Metrics
    """)
    return


@app.cell
def _(lstm_predictions, lstm_scaler, lstm_test_normalized, np, plt):
    lstm_test_predictions_mw = lstm_scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

    # we make sure to only print hours where we could look back 2 years (the lag features)
    lstm_min_len = min(len(lstm_test_normalized), len(lstm_test_predictions_mw))
    lstm_y_test_mw = lstm_scaler.inverse_transform(lstm_test_normalized[:lstm_min_len, 0].reshape(-1, 1))
    lstm_test_predictions_mw = lstm_test_predictions_mw[:lstm_min_len]

    # plot first week
    plt.figure(figsize=(15, 6))
    plt.plot(lstm_y_test_mw[:168], label='Actual', color='blue')
    plt.plot(lstm_test_predictions_mw[:168], label='24h Chunk Prediction', color='red', linestyle='--')
    plt.title("24-Hour Chunk Prediction (First Week)")
    plt.legend()
    plt.show()

    # plot first month
    plt.figure(figsize=(15, 6))
    plt.plot(lstm_y_test_mw[:720], label='Actual', color='blue')
    plt.plot(lstm_test_predictions_mw[:720], label='24h Chunk Prediction', color='red', linestyle='--')
    plt.title("24-Hour Chunk Prediction (First Month - Seasonality Check)")
    plt.legend()
    plt.show()

    # plot first year
    plt.figure(figsize=(15, 6))
    plt.plot(lstm_y_test_mw[:24*365], label='Actual', color='blue')
    plt.plot(lstm_test_predictions_mw[:24*365], label='24h Chunk Prediction', color='green', linestyle='--')
    plt.title("24-Hour Chunk Prediction (First Month - Seasonality Check)")
    plt.legend()
    plt.show()
    return


@app.cell
def _(
    lstm_df_small,
    lstm_predictions,
    lstm_scaler,
    lstm_test_normalized,
    lstm_train_size,
    mean_squared_error,
    np,
):
    lstm_test_predictions_mw_flat = lstm_scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()

    lstm_min_len_flat = min(len(lstm_test_normalized), len(lstm_test_predictions_mw_flat))
    lstm_y_test_mw_flat = lstm_scaler.inverse_transform(lstm_test_normalized[:lstm_min_len_flat, 0].reshape(-1, 1)).flatten()
    lstm_test_predictions_mw_flat = lstm_test_predictions_mw_flat[:lstm_min_len_flat]

    lstm_actuals = lstm_y_test_mw_flat
    lstm_preds = lstm_test_predictions_mw_flat

    lstm_horizons = {
        '24 Hours (1 Day)': 24,
        '168 Hours (1 Week)': 168,
        '720 Hours (1 Month)': 720,
        '8760 Hours (1 Year)': 8760
    }
    lstm_full_actuals = lstm_df_small['mw'].values

    print("=== Horizon-Based Sequential Evaluation ===\n")

    for lstm_name, lstm_horizon_h in lstm_horizons.items():
        # safety check
        if lstm_horizon_h > len(lstm_y_test_mw_flat):
            print(f"[{lstm_name}] SKIP: Not enough data in the test set. (Test set size: {len(lstm_y_test_mw_flat)}h, Needed: {lstm_horizon_h}h)\n")
            continue

        lstm_actuals_h = lstm_y_test_mw_flat[:lstm_horizon_h]
        lstm_preds_h = lstm_test_predictions_mw_flat[:lstm_horizon_h].flatten()
        lstm_baseline_h = lstm_full_actuals[lstm_train_size - lstm_horizon_h : lstm_train_size]

        lstm_model_rmse = np.sqrt(mean_squared_error(lstm_actuals_h, lstm_preds_h))
        lstm_model_mape = np.mean(np.abs((lstm_actuals_h - lstm_preds_h) / lstm_actuals_h)) * 100

        lstm_baseline_rmse = np.sqrt(mean_squared_error(lstm_actuals_h, lstm_baseline_h))
        lstm_baseline_mape = np.mean(np.abs((lstm_actuals_h - lstm_baseline_h) / lstm_actuals_h)) * 100

        print(f"--- {lstm_name} ---")
        print(f"Model RMSE:    {lstm_model_rmse:.2f} MW  |  MAPE: {lstm_model_mape:.2f}%")
        print(f"Baseline RMSE: {lstm_baseline_rmse:.2f} MW  |  MAPE: {lstm_baseline_mape:.2f}%")

        if lstm_model_rmse < lstm_baseline_rmse:
            lstm_diff = lstm_baseline_rmse - lstm_model_rmse
            print(f"SUCCESS: Model BEAT baseline by {lstm_diff:.2f} MW")
        else:
            lstm_diff = lstm_model_rmse - lstm_baseline_rmse
            print(f"FAILURE: Model LOST to baseline by {lstm_diff:.2f} MW")
        print("-" * 45)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Expanding Window
    """)
    return


@app.cell
def _(
    DataLoader,
    LSTMModel,
    MinMaxScaler,
    TensorDataset,
    energy_data,
    lstm_create_chunk_sequences,
    mean_absolute_percentage_error,
    mean_squared_error,
    nn,
    np,
    os,
    plt,
    random,
    torch,
):
    validation_energy_data = energy_data.copy()
    def validation_set_seed(seed_value=42):
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    validation_set_seed(42)

    validation_energy_data['year'] = validation_energy_data['date'].dt.year

    validation_df_small = validation_energy_data[(validation_energy_data['year'] <= 2017) & (validation_energy_data['year'] >= 2007)].copy()

    validation_day_year = validation_df_small['date'].dt.dayofyear
    validation_df_small['sin_year'] = np.sin(2 * np.pi * validation_day_year / 365)
    validation_df_small['cos_year'] = np.cos(2 * np.pi * validation_day_year / 365)
    validation_df_small['sin_hour'] = np.sin(2 * np.pi * validation_df_small['date'].dt.hour / 24)
    validation_df_small['cos_hour'] = np.cos(2 * np.pi * validation_df_small['date'].dt.hour / 24)
    validation_df_small['sin_day'] = np.sin(2 * np.pi * validation_df_small['date'].dt.dayofweek / 7)
    validation_df_small['cos_day'] = np.cos(2 * np.pi * validation_df_small['date'].dt.dayofweek / 7)

    validation_feature_cols = ['mw', 'sin_year', 'cos_year', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day']

    validation_start_year = 2007
    validation_initial_train_years = 3
    validation_num_folds = 5
    validation_fold_metrics = []

    for validation_fold in range(validation_num_folds):
        validation_train_end_year = validation_start_year + validation_initial_train_years + validation_fold
        validation_test_year = validation_train_end_year

        print(f"\n{'='*50}")
        print(f"FOLD {validation_fold + 1}: Train -> {validation_start_year}-{validation_train_end_year-1} | Test -> {validation_test_year}")
        print(f"{'='*50}")

        validation_df_train = validation_df_small[validation_df_small['year'] < validation_test_year].copy()
        validation_df_test = validation_df_small[validation_df_small['year'] == validation_test_year].copy()

        validation_train_raw = validation_df_train[validation_feature_cols].values
        validation_test_raw = validation_df_test[validation_feature_cols].values

        validation_scaler = MinMaxScaler()
        validation_scaler.fit(validation_train_raw[:, 0].reshape(-1, 1))

        validation_train_normalized = validation_train_raw.copy()
        validation_test_normalized = validation_test_raw.copy()
        validation_train_normalized[:, 0] = validation_scaler.transform(validation_train_raw[:, 0].reshape(-1, 1)).flatten()
        validation_test_normalized[:, 0] = validation_scaler.transform(validation_test_raw[:, 0].reshape(-1, 1)).flatten()

        validation_X_train, validation_y_train = lstm_create_chunk_sequences(validation_train_normalized)

        validation_X_train_tensor = torch.FloatTensor(validation_X_train)
        validation_y_train_tensor = torch.FloatTensor(validation_y_train)

        validation_train_dataset = TensorDataset(validation_X_train_tensor, validation_y_train_tensor)
        validation_train_loader = DataLoader(validation_train_dataset, batch_size=64, shuffle=True)

        validation_model = LSTMModel()
        validation_optimizer = torch.optim.Adam(validation_model.parameters(), lr=0.001)
        validation_criterion = nn.MSELoss()

        print(f"Training on {len(validation_X_train)} samples...")
        for validation_epoch in range(5):
            validation_model.train()
            validation_epoch_loss = 0
            for validation_batch_X, validation_batch_y in validation_train_loader:
                validation_optimizer.zero_grad()
                validation_loss = validation_criterion(validation_model(validation_batch_X), validation_batch_y)
                validation_loss.backward()
                validation_optimizer.step()
                validation_epoch_loss += validation_loss.item()
            print(f"  Epoch {validation_epoch+1} Loss: {validation_epoch_loss/len(validation_train_loader):.5f}")

        validation_model.eval()
        validation_history = [validation_row for validation_row in validation_train_normalized]
        validation_predictions = []

        validation_lookback_buffer = validation_train_normalized[-(24*365*2):]
        validation_test_with_history = np.concatenate((validation_lookback_buffer, validation_test_normalized))

        validation_total_test_hours = len(validation_test_normalized)
        validation_num_chunks = int(np.ceil(validation_total_test_hours / 24))
        validation_future_time_features = validation_test_normalized[:, 1:]

        print(f"Predicting {validation_test_year} in {validation_num_chunks} chunks...")
        with torch.no_grad():
            for validation_chunk_idx in range(validation_num_chunks):
                validation_t = len(validation_history)
                validation_indices = []

                for validation_h in range(3, 0, -1): validation_indices.append(validation_t - validation_h)
                validation_indices.append(validation_t - 24)
                validation_indices.append(validation_t - (24 * 7))
                validation_indices.append(validation_t - (24 * 365))
                validation_indices.append(validation_t - (24 * 365 * 2))

                validation_seq_data = np.array([validation_history[validation_idx] for validation_idx in validation_indices])
                validation_current_input = torch.FloatTensor(validation_seq_data).unsqueeze(0)
                validation_chunk_pred = validation_model(validation_current_input).numpy()[0]

                validation_predictions.extend(validation_chunk_pred)

                validation_start_t = validation_chunk_idx * 24
                validation_end_t = validation_start_t + 24
                validation_current_chunk_times = validation_future_time_features[validation_start_t : validation_end_t]

                if len(validation_current_chunk_times) < 24:
                    validation_predictions = validation_predictions[:-(24 - len(validation_current_chunk_times))]
                    break

                validation_new_rows = np.column_stack((validation_chunk_pred, validation_current_chunk_times))
                for validation_row in validation_new_rows:
                    validation_history.append(validation_row)

        validation_test_predictions_mw = validation_scaler.inverse_transform(np.array(validation_predictions).reshape(-1, 1))

        validation_min_len = min(len(validation_test_normalized), len(validation_test_predictions_mw))
        validation_y_test_mw = validation_scaler.inverse_transform(validation_test_normalized[:validation_min_len, 0].reshape(-1, 1))
        validation_test_predictions_mw = validation_test_predictions_mw[:validation_min_len]

        validation_rmse = np.sqrt(mean_squared_error(validation_y_test_mw, validation_test_predictions_mw))
        validation_mape = mean_absolute_percentage_error(validation_y_test_mw, validation_test_predictions_mw)

        print(f"--> Fold {validation_fold+1} ({validation_test_year}) Results: RMSE: {validation_rmse:.2f} MW | MAPE: {validation_mape:.4f}")
        validation_fold_metrics.append({
            'fold': validation_fold + 1,
            'year': validation_test_year,
            'rmse': validation_rmse,
            'mape': validation_mape,
            'train_dates': validation_df_train['date'].values,
            'train_actuals': validation_df_train['mw'].values,
            'test_dates': validation_df_test['date'].values[:validation_min_len],
            'test_actuals': validation_y_test_mw.flatten(),
            'test_preds': validation_test_predictions_mw.flatten()
        })

    print(f"\n{'='*50}")
    print("FINAL EXPANDING WINDOW RESULTS")
    print(f"{'='*50}")
    validation_avg_rmse = np.mean([validation_m['rmse'] for validation_m in validation_fold_metrics])
    validation_avg_mape = np.mean([validation_m['mape'] for validation_m in validation_fold_metrics])

    for validation_m in validation_fold_metrics:
        print(f"Year {validation_m['year']} - RMSE: {validation_m['rmse']:.2f} | MAPE: {validation_m['mape']:.4f}")
    print("-" * 50)
    print(f"AVERAGE RMSE: {validation_avg_rmse:.2f} MW")
    print(f"AVERAGE MAPE: {validation_avg_mape:.4f}")


    # nice plots we like
    validation_fig, validation_axes = plt.subplots(validation_num_folds, 1, figsize=(15, 20), sharex=True)
    validation_fig.suptitle("Expanding Window Validation Forecasts", fontsize=18, y=0.98)

    validation_forecast_colors = ['tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']

    for validation_i, validation_m in enumerate(validation_fold_metrics):
        validation_ax = validation_axes[validation_i]

        # plot history
        validation_ax.plot(validation_m['train_dates'], validation_m['train_actuals'],
                label='Historical Training Data', color='#1f77b4', alpha=0.8)

        # plot actual future
        validation_ax.plot(validation_m['test_dates'], validation_m['test_actuals'],
                label='Actual Energy Usage', color='black', alpha=0.2, linestyle='-')

        # plot predictions
        validation_ax.plot(validation_m['test_dates'], validation_m['test_preds'],
                label=f"Predicted {validation_m['year']}", color=validation_forecast_colors[validation_i % len(validation_forecast_colors)], linewidth=2)

        validation_ax.set_title(f"Fold {validation_m['fold']} | History: 2007-{validation_m['year']-1} -> Predicting: {validation_m['year']} (RMSE: {validation_m['rmse']:.0f} MW)")
        validation_ax.legend(loc='upper left')
        validation_ax.grid(True, alpha=0.3)
        validation_ax.set_ylabel("Megawatts (MW)")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    return validation_fold_metrics, validation_num_folds


@app.cell
def _(pd, plt, validation_fold_metrics, validation_num_folds):
    smoothed_fig, smoothed_axes = plt.subplots(validation_num_folds, 1, figsize=(15, 20), sharex=True)
    smoothed_fig.suptitle("Expanding Window Validation Forecasts (1-Week Rolling Average)", fontsize=18, y=0.98)
    smoothed_forecast_colors = ['tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']
    smoothed_window_size = 168

    for smoothed_i, smoothed_m in enumerate(validation_fold_metrics):
        smoothed_ax = smoothed_axes[smoothed_i]

        def smoothed_apply_smoothing(data_array):
            return pd.Series(data_array).rolling(window=smoothed_window_size, min_periods=1).mean().values

        smoothed_smooth_train = smoothed_apply_smoothing(smoothed_m['train_actuals'])
        smoothed_smooth_actuals = smoothed_apply_smoothing(smoothed_m['test_actuals'])
        smoothed_smooth_preds = smoothed_apply_smoothing(smoothed_m['test_preds'])

        # plot history
        smoothed_ax.plot(smoothed_m['train_dates'], smoothed_smooth_train,
                label='Historical Training Data', color='#1f77b4', linewidth=1.5)

        # plot actuals
        smoothed_ax.plot(smoothed_m['test_dates'], smoothed_smooth_actuals,
                label='Actual Energy Usage', color='black', alpha=0.4, linestyle='-', linewidth=1.5)

        # plot predictions
        smoothed_ax.plot(smoothed_m['test_dates'], smoothed_smooth_preds,
                label=f"Predicted {smoothed_m['year']}", color=smoothed_forecast_colors[smoothed_i % len(smoothed_forecast_colors)], linewidth=2.5)

        smoothed_ax.set_title(f"Fold {smoothed_m['fold']} | History: 2007-{smoothed_m['year']-1} -> Predicting: {smoothed_m['year']} (RMSE: {smoothed_m['rmse']:.0f} MW)")
        smoothed_ax.legend(loc='upper left')
        smoothed_ax.grid(True, alpha=0.3)
        smoothed_ax.set_ylabel("Megawatts (MW)")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Final task: compare with Timestral-5M

    So far, our models learned from this electricity dataset. A **pretrained model**
    starts with weights learned from other data. Here we will compare our LSTM with
    [Mistral's Timestral-5M](https://huggingface.co/mistralai/Timestral-5M), a time-series
    transformer with about five million parameters.

    - **Zero-shot:** give the pretrained model past measurements and ask for a forecast.
      We do not update its weights on our electricity data.
    - **Fine-tuned:** start from a separate copy of those same pretrained weights and
      update them using windows from our training period only.

    We will forecast the next **24 hours** at each of the first **14 daily origins**
    in the existing test period. Each new day may use measurements observed before
    that day's forecast. We never give a model the actual consumption within its
    24-hour forecast window. Fine-tuning never uses test measurements.

    This is a rolling day-ahead experiment. The earlier plots recursively forecast a
    much longer future, so their error numbers are not directly comparable. We will
    re-evaluate the LSTM on these same daily targets and add a "same hour yesterday"
    baseline. The LSTM keeps its original lag/time features; Timestral receives the
    most recent 672 hourly consumption values (four weeks).

    **Before starting:** install the optional packages and open the workbook with:

    ```sh
    uv run --locked --project bootcamp_9 --extra timestral python bootcamp_9/start.py
    ```

    Add `--solution` for the worked version. This optional extra installs Mistral's
    pinned Git repository, so Git and internet access are needed on first setup.
    The approximately 21 MB model download is cached locally. In VS Code, run the
    same command with `--setup`, then keep using `bootcamp_9/.venv` as the kernel.
    Click the button below when you are ready; opening the notebook alone does not
    start this extra download or fine-tuning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## A. Forecast without training

    Use model ID **"mistralai/Timestral-5M"** and a forecast horizon of **24**.
    The supplied helper loads a fixed checkpoint on the CPU. Its `forecast` method
    returns several quantiles; `.median` selects one central prediction per hour.

    Fill in the two settings below, then click **Run Timestral task**. The data setup
    reuses the original chronological split. Windows with missing or duplicated
    hours are excluded from fine-tuning; the chosen evaluation period is checked
    for hourly continuity.
    """)
    return


@app.cell
def _(mo):
    timestral_run = mo.ui.run_button(label="Run Timestral task")
    timestral_run
    return (timestral_run,)


@app.cell
def _():
    timestral_model_id = "mistralai/Timestral-5M"
    timestral_horizon = 24
    return timestral_horizon, timestral_model_id


@app.cell
def _(Path, mo, timestral_horizon, timestral_model_id, timestral_run):
    mo.stop(not timestral_run.value)
    mo.stop(
        timestral_model_id is None or timestral_horizon is None,
        mo.md('Fill in the Timestral model ID and the 24-hour horizon above.'),
    )
    import importlib.util as _importlib_util
    import sys as _sys
    mo.stop(
        _importlib_util.find_spec("timestral") is None,
        mo.md("Install the optional extra: `uv run --locked --project bootcamp_9 --extra timestral python bootcamp_9/start.py --setup`, then restart this kernel."),
    )
    timestral_root = Path(mo.notebook_location()).resolve().parent
    _spec = _importlib_util.spec_from_file_location(
        "bootcamp_nine_timestral_task", timestral_root / "timestral_task.py"
    )
    timestral_tools = _importlib_util.module_from_spec(_spec)
    _sys.modules[_spec.name] = timestral_tools
    _spec.loader.exec_module(timestral_tools)
    return timestral_root, timestral_tools


@app.cell
def _(lstm_df_small, lstm_train_size, timestral_horizon, timestral_tools):
    timestral_data = timestral_tools.prepare_comparison(
        lstm_df_small, lstm_train_size,
        context_length=672, horizon=timestral_horizon, days=14,
    )
    print(f"Training measurements: {len(timestral_data.train_values):,}")
    print(f"Evaluation: {timestral_data.actuals.shape[0]} forecasts × {timestral_data.horizon} hours")
    print(f"First forecast starts at {timestral_data.dates[0, 0]}")
    return (timestral_data,)


@app.cell
def _(timestral_data, timestral_model_id, timestral_root, timestral_tools):
    timestral_pretrained = timestral_tools.load_pretrained(
        cache_dir=timestral_root / ".cache" / "timestral",
        model_id=timestral_model_id,
    )
    timestral_zero_shot = timestral_tools.forecast_days(timestral_pretrained, timestral_data)
    print(f"Loaded {sum(p.numel() for p in timestral_pretrained.model.parameters()):,} parameters")
    print("Zero-shot forecast shape:", timestral_zero_shot.shape)
    return timestral_pretrained, timestral_zero_shot


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## B. Adapt the pretrained model to our training data

    Use **40** optimizer updates for this short experiment. The helper below copies
    the pretrained model before training, so our zero-shot model stays unchanged.
    It samples batches of past-to-future windows entirely inside the training period
    and updates all model weights with a small learning rate (`1e-5`).

    The last 14 days of the original training period are reserved for validation;
    their targets never enter an optimizer update. Every 10 updates, we evaluate
    the same 24-hour forecasting task on that validation period and keep the best
    trained checkpoint by validation RMSE. The zero-shot validation score is shown
    too: if it wins, prefer the unadapted model. We do not select a checkpoint
    using the test scores below.

    Timestral predicts 21 quantiles. The supplied training loop uses **pinball loss**:
    for a quantile such as 0.9, under-predicting is penalized more than over-predicting.
    The median quantile uses a symmetric absolute-error loss. You do not need to
    implement that loss for this task; inspect `timestral_task.py` if you want to
    follow the PyTorch forward, backward and optimizer steps.

    The public Mistral package provides inference code. This is a small educational
    fine-tuning loop around its differentiable model, rather than the Mistral Forge
    training pipeline. Forty updates demonstrate adaptation; they do not guarantee
    an improvement. The loop matches the public forecaster's symmetric averaging
    and non-negative clipping. Choose settings using training/validation data,
    not this test table.
    """)
    return


@app.cell
def _():
    timestral_fine_tune_steps = 40
    return (timestral_fine_tune_steps,)


@app.cell
def _(
    mo,
    timestral_data,
    timestral_fine_tune_steps,
    timestral_pretrained,
    timestral_tools,
):
    mo.stop(
        timestral_fine_tune_steps is None,
        mo.md('Set the number of fine-tuning updates to 40 above.'),
    )
    timestral_adapted, timestral_losses = timestral_tools.fine_tune(
        timestral_pretrained, timestral_data,
        steps=timestral_fine_tune_steps, batch_size=8, learning_rate=1e-5,
    )
    timestral_fine_tuned = timestral_tools.forecast_days(timestral_adapted, timestral_data)
    print(f"Selected update: {timestral_losses.selected_step}")
    print(f"Validation RMSE: zero-shot {timestral_losses.zero_shot_validation_rmse:.2f} MW; "
          f"fine-tuned {timestral_losses.selected_validation_rmse:.2f} MW")
    return timestral_fine_tuned, timestral_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## C. Compare and explain

    The table evaluates every model on the same 14 × 24 held-out hourly values.
    RMSE and MAE are in **MW**; MAPE is a percentage. Lower is better.

    1. Does zero-shot Timestral beat our trained LSTM and the seasonal baseline?
    2. Did fine-tuning help? If it became worse, what could you change using a
       validation period inside the training data?
    3. What does "zero-shot" mean here? Does the model still receive past observations?
    4. Does this comparison isolate architecture? Consider pretraining, history length,
       calendar features and training budget before attributing a difference to size.
    5. Would the ranking hold in another season? Plan more forecast origins before
       making a general claim about which model is best.

    **Interpretation:** zero-shot here means no local gradient updates. Mistral's
    release describes synthetic and real pretraining data but does not establish
    whether this public AEP series was included. Treat this as a classroom comparison,
    not proof of performance on data unseen during pretraining.
    """)
    return


@app.cell
def _(
    lstm_model,
    lstm_scaler,
    lstm_test_normalized,
    lstm_train_normalized,
    mo,
    timestral_data,
    timestral_fine_tuned,
    timestral_tools,
    timestral_zero_shot,
):
    timestral_lstm_predictions = timestral_tools.forecast_lstm(
        lstm_model, lstm_scaler, lstm_train_normalized,
        lstm_test_normalized, timestral_data,
    )
    timestral_scores = timestral_tools.score_forecasts(timestral_data.actuals, {
        "Same hour yesterday": timestral_data.seasonal_naive,
        "Our LSTM": timestral_lstm_predictions,
        "Timestral-5M: zero-shot": timestral_zero_shot,
        "Timestral-5M: fine-tuned": timestral_fine_tuned,
    })
    mo.Html(timestral_scores.round(2).to_html(index=False))
    return (timestral_lstm_predictions,)


@app.cell
def _(
    plt,
    timestral_data,
    timestral_fine_tuned,
    timestral_lstm_predictions,
    timestral_zero_shot,
):
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _hours = timestral_data.dates.ravel()[:168]
    _ax.plot(_hours, timestral_data.actuals.ravel()[:168], color="black", label="Actual", linewidth=2)
    for _name, _prediction in [
        ("Same hour yesterday", timestral_data.seasonal_naive),
        ("Our LSTM", timestral_lstm_predictions),
        ("Timestral zero-shot", timestral_zero_shot),
        ("Timestral fine-tuned", timestral_fine_tuned),
    ]:
        _ax.plot(_hours, _prediction.ravel()[:168], label=_name, alpha=0.8)
    _ax.set_title("Day-ahead forecasts: first week of the comparison")
    _ax.set_ylabel("Consumption (MW)")
    _ax.set_xlabel("Forecasted time")
    _ax.legend()
    _fig.autofmt_xdate()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(plt, timestral_losses):
    _fig, (_train_ax, _val_ax) = plt.subplots(2, 1, figsize=(8, 6))
    _train_ax.plot(range(1, len(timestral_losses) + 1), timestral_losses)
    _train_ax.set_ylabel("Training pinball loss")
    _train_ax.set_title("Fine-tuning: learning versus generalisation")
    _steps = [_record["step"] for _record in timestral_losses.validation]
    _scores = [_record["rmse_mw"] for _record in timestral_losses.validation]
    _val_ax.plot(_steps, _scores, marker="o", label="Trained checkpoints")
    _val_ax.axhline(timestral_losses.zero_shot_validation_rmse, color="black",
                    linestyle="--", label="Zero-shot validation baseline")
    _val_ax.set_ylabel("Validation RMSE (MW)")
    _val_ax.set_xlabel("Optimizer update")
    _val_ax.legend()
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
