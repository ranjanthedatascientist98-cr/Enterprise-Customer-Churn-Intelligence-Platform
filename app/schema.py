from pydantic import BaseModel


class CustomerData(BaseModel):
    Tenure: int
    WarehouseToHome: int
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    DaySinceLastOrder: int
    CashbackAmount: float