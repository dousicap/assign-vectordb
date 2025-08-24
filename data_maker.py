#!/usr/bin/env python3
import random
from datetime import datetime, timedelta
import csv
import zipfile
import argparse
import os

def random_date(start, end):
    """Return a random date between start and end (inclusive)."""
    if end < start:
        start, end = end, start
    delta = end - start
    days = delta.days
    if days <= 0:
        return start
    return start + timedelta(days=random.randint(0, days))

def clamp(n, lo, hi):
    return max(lo, min(n, hi))

def generate_dataset(n=2000, seed=42):
    random.seed(seed)

    genders = ['Male', 'Female', 'Other']
    regions = ['North', 'South', 'East', 'West', 'Central']
    employment = ['Employed', 'Self-employed', 'Unemployed', 'Student', 'Retired']
    loan_types = ['Mortgage', 'Auto', 'Personal', 'Student', 'Credit Card']

    rate_ranges = {
        'Mortgage': (3.0, 6.5),
        'Auto': (3.0, 6.5),
        'Personal': (6.0, 18.0),
        'Student': (4.0, 9.0),
        'Credit Card': (12.0, 25.0),
    }

    amount_ranges = {
        'Mortgage': (100_000, 2_000_000),
        'Auto': (10_000, 80_000),
        'Personal': (1_000, 50_000),
        'Student': (5_000, 150_000),
        'Credit Card': (1_000, 50_000),
    }

    term_options = [12, 24, 36, 60, 120, 180, 240, 360]

    now = datetime.now()
    start_date = now - timedelta(days=3650)  # up to ~10 years back

    header = [
        'record_id', 'customer_id', 'age', 'gender', 'region', 'employment_status',
        'annual_income', 'credit_score', 'loan_type', 'loan_amount',
        'annual_interest_rate', 'loan_term_months', 'loan_status', 'open_date',
        'last_payment_date', 'last_payment_amount', 'balance_remaining',
        'currency', 'debt_to_income_ratio', 'risk_band'
    ]

    data = []
    for i in range(1, n + 1):
        record_id = i
        customer_id = f"CUST{100000 + i:06d}"
        age = random.randint(18, 80)
        gender = random.choice(genders)
        region = random.choice(regions)
        employment_status = random.choice(employment)

        annual_income = random.randint(20_000, 350_000)
        credit_score = int(random.normalvariate(700, 60))
        credit_score = clamp(credit_score, 300, 850)

        loan_type = random.choice(loan_types)
        loan_amount = random.randint(*amount_ranges[loan_type])

        rate_min, rate_max = rate_ranges[loan_type]
        annual_interest_rate = round(random.uniform(rate_min, rate_max), 2)

        loan_term_months = random.choice(term_options)

        # Simple amortization estimate
        monthly_rate = annual_interest_rate / 12.0 / 100.0
        n_payments = loan_term_months
        if monthly_rate > 0:
            monthly_payment = loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-n_payments))
        else:
            monthly_payment = loan_amount / n_payments

        open_date = random_date(start_date, now)

        months_paid = random.randint(0, loan_term_months)
        balance_remaining = max(0.0, loan_amount - monthly_payment * months_paid)

        last_payment_days = random.randint(0, max(1, months_paid * 30))
        last_payment_date = open_date + timedelta(days=last_payment_days)
        if last_payment_date > now:
            last_payment_date = now
        last_payment_amount = max(0.0, monthly_payment * random.uniform(0.7, 1.1))

        status_weights = {'Active': 0.75, 'Closed': 0.15, 'Delinquent': 0.07, 'Default': 0.03}
        loan_status = random.choices(list(status_weights.keys()), weights=list(status_weights.values()), k=1)[0]

        if loan_status == 'Active':
            payment_status = random.choices(['On-time', 'Late', 'Missed'], weights=[0.70, 0.20, 0.10], k=1)[0]
        elif loan_status == 'Delinquent':
            payment_status = random.choices(['On-time', 'Late', 'Missed'], weights=[0.30, 0.40, 0.30], k=1)[0]
        else:
            payment_status = 'On-time' if loan_status == 'Closed' else 'Missed'

        currency = 'USD'
        debt_to_income_ratio = round((monthly_payment) / (annual_income / 12.0), 4)

        if credit_score >= 750:
            risk_band = 'Low'
        elif credit_score >= 650:
            risk_band = 'Medium'
        else:
            risk_band = 'High'

        row = {
            'record_id': record_id,
            'customer_id': customer_id,
            'age': age,
            'gender': gender,
            'region': region,
            'employment_status': employment_status,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'loan_type': loan_type,
            'loan_amount': round(loan_amount, 2),
            'annual_interest_rate': annual_interest_rate,
            'loan_term_months': loan_term_months,
            'loan_status': loan_status,
            'open_date': open_date.strftime('%Y-%m-%d'),
            'last_payment_date': last_payment_date.strftime('%Y-%m-%d'),
            'last_payment_amount': round(last_payment_amount, 2),
            'balance_remaining': round(balance_remaining, 2),
            'currency': currency,
            'debt_to_income_ratio': round(debt_to_income_ratio, 4),
            'risk_band': risk_band
        }

        data.append(row)

    return header, data

def save_csv(filename, header, rows):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[h] for h in header])

def save_zip(csv_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=os.path.basename(csv_path))

def main():
    parser = argparse.ArgumentParser(description="Generate random financial data (CSV) and zip it (around 2000 records).")
    parser.add_argument('-n', '--num', type=int, default=2000, help='Number of records (default: 2000)')
    parser.add_argument('-o', '--output', default='finance_data', help='Output base name (without extension)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    header, rows = generate_dataset(n=max(1, int(args.num)), seed=args.seed)

    csv_path = f"{args.output}.csv"
    zip_path = f"{args.output}.zip"

    save_csv(csv_path, header, rows)
    save_zip(csv_path, zip_path)

    print(f"Generated: {csv_path} and {zip_path}")

if __name__ == "__main__":
    main()