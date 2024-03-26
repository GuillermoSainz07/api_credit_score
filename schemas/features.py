from pydantic import BaseModel


class Features(BaseModel):
    month_salary: float
    bank_accounts: float
    num_credit_card: float
    interest_rate: float
    delay_date: float
    credit_inquiries: float
    credit_utilization: float
    emi_per_month: float
    age: float
    annual_income: float
    num_loan: float
    delay_payment: float
    chaged_limit: float
    outstanding_debt: float
    amount_invested: float
    month_balance: float
    occupation: str
    history_age: float
    paymen_min: str
    payment_behaviour: str
    payday_loan: float
    personal_loan: float
    mortgage_loan: float
    student_loan: float
    auto_loan: float
    credit_builder_loan: float
    home_equity_loan: float
    debt_cons_loan: float



