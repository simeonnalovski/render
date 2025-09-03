from datetime import date, datetime
from typing import  List
from pydantic import BaseModel, TypeAdapter, ValidationError,validator
import pandas as pd


class FinancialRecord(BaseModel):
    date: date
    symbol: str
    reportedCurrency: str
    cik: int
    filingDate: date
    acceptedDate: datetime
    fiscalYear: int
    period: str
    revenue: int
    costOfRevenue: int
    grossProfit: int
    researchAndDevelopmentExpenses: int
    generalAndAdministrativeExpenses: int
    sellingAndMarketingExpenses: int
    sellingGeneralAndAdministrativeExpenses: int
    otherExpenses: int
    operatingExpenses: int
    costAndExpenses: int
    netInterestIncome: int
    interestIncome: int
    interestExpense: int
    depreciationAndAmortization: int
    ebitda: int
    ebit: int
    nonOperatingIncomeExcludingInterest: int
    operatingIncome: int
    totalOtherIncomeExpensesNet: int
    incomeBeforeTax: int
    incomeTaxExpense: int
    netIncomeFromContinuingOperations: int
    netIncomeFromDiscontinuedOperations: int
    otherAdjustmentsToNetIncome: int
    netIncome: int
    netIncomeDeductions: int
    bottomLineNetIncome: int
    eps: float
    epsDiluted: float
    weightedAverageShsOut: int
    weightedAverageShsOutDil: int

    @validator('revenue')
    def revenue_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Revenue must be a non-negative value.')
        return v


# Validator for list of records
CSVValidator = TypeAdapter(List[FinancialRecord])


def validate_dataframe(df: pd.DataFrame) -> bool:
    try:
        df = df.where(pd.notnull(df), None)  # replace NaN with None
        records = df.to_dict(orient="records")
        CSVValidator.validate_python(records)  # will raise if invalid
        return True
    except ValidationError as e:
        return False


