#FinalProjectCustomerConversionPrediction

#Start with importing libraries wich we will use in the project

import pandas as pd
import numpy as np
import seaborn as sn           
import matplotlib.pyplot as plt
%matplotlib inline

For ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Import Data
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "79528170",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(r\"C:\\Users\\kuldi\\Downloads\\GUVI - IITM Final Project\\Customer Conversion Prediction - Customer Conversion Prediction.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "156ac29c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education_qual</th>\n",
       "      <th>call_type</th>\n",
       "      <th>day</th>\n",
       "      <th>mon</th>\n",
       "      <th>dur</th>\n",
       "      <th>num_calls</th>\n",
       "      <th>prev_outcome</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>58</td>\n",
       "      <td>management</td>\n",
       "      <td>married</td>\n",
       "      <td>tertiary</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>261</td>\n",
       "      <td>1</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>44</td>\n",
       "      <td>technician</td>\n",
       "      <td>single</td>\n",
       "      <td>secondary</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>151</td>\n",
       "      <td>1</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>33</td>\n",
       "      <td>entrepreneur</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>76</td>\n",
       "      <td>1</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>47</td>\n",
       "      <td>blue-collar</td>\n",
       "      <td>married</td>\n",
       "      <td>unknown</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>92</td>\n",
       "      <td>1</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>33</td>\n",
       "      <td>unknown</td>\n",
       "      <td>single</td>\n",
       "      <td>unknown</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>198</td>\n",
       "      <td>1</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age           job  marital education_qual call_type  day  mon  dur  \\\n",
       "0   58    management  married       tertiary   unknown    5  may  261   \n",
       "1   44    technician   single      secondary   unknown    5  may  151   \n",
       "2   33  entrepreneur  married      secondary   unknown    5  may   76   \n",
       "3   47   blue-collar  married        unknown   unknown    5  may   92   \n",
       "4   33       unknown   single        unknown   unknown    5  may  198   \n",
       "\n",
       "   num_calls prev_outcome   y  \n",
       "0          1      unknown  no  \n",
       "1          1      unknown  no  \n",
       "2          1      unknown  no  \n",
       "3          1      unknown  no  \n",
       "4          1      unknown  no  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35466a08",
   "metadata": {},
   "source": [
    "Let's check the features present in our data and then we will look at data shape and data types."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "dd0ae643",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['age', 'job', 'marital', 'education_qual', 'call_type', 'day', 'mon',\n",
       "       'dur', 'num_calls', 'prev_outcome', 'y'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "58475682",
   "metadata": {},
   "source": [
    "We have 17 independent variables and 1 target variable. Target variable = 'y'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "99874be7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(45211, 11)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b1c8c1b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age                int64\n",
       "job               object\n",
       "marital           object\n",
       "education_qual    object\n",
       "call_type         object\n",
       "day                int64\n",
       "mon               object\n",
       "dur                int64\n",
       "num_calls          int64\n",
       "prev_outcome      object\n",
       "y                 object\n",
       "dtype: object"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "67ef68c7",
   "metadata": {},
   "source": [
    "e can see there are two format of data types:\n",
    "\n",
    "1. **object**: Object format means variables are categorical. Categorical variables in our dataset are: job, marital, education_qual, call_type, mon, prev_outcome, y.\n",
    "\n",
    "2. **int64**: It represents the integer variables. Integer variables in our dataset are: age, day, dur, num_calls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "82cff249",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "32d798b7",
   "metadata": {},
   "source": [
    "## Univariate Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5e10b327",
   "metadata": {},
   "source": [
    "Let's look at our target variable 'y' first. It is catagorical variable. We'll see it's distribution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "536cf8ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "no     39922\n",
       "yes     5289\n",
       "Name: y, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['y'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "991bb3d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "no     0.883015\n",
       "yes    0.116985\n",
       "Name: y, dtype: float64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# See the % distribution of the \"y\"\n",
    "data['y'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b1ab97a6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYQAAAEACAYAAACznAEdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAVd0lEQVR4nO3df4xd5X3n8fcndpp420L4MSDH46xZsKoaujGLZVkif2TrKrhptSYVrAZpg3fXWkfI0aZSpRVUu5vkD0uhUuoV0mLVKSyGdmssmq6tCLqwplFVLWt3kqUYQyxmQwKDvXgSCHG0xa3Nd/+4z0jXw2Xmzow91zDvl3R0z/2e5znzHHk0n3vOc65PqgpJkj406AFIki4OBoIkCTAQJEmNgSBJAgwESVJjIEiSAFg66AHM1ZVXXlmrVq0a9DAk6X3lO9/5zo+qaqjXtvdtIKxatYrR0dFBD0OS3leS/PC9tnnJSJIEGAiSpMZAkCQBBoIkqek7EJIsSfK/k3yrvb88yVNJXmqvl3W1vSfJWJJjSW7pqt+U5Ejbdl+StPpHkjza6oeSrDqPxyhJ6sNszhC+BLzY9f5u4GBVrQYOtvckWQOMANcDm4D7kyxpfXYB24DVbdnU6luBN6vqOmAncO+cjkaSNGd9BUKSYeA3gD/sKm8G9rT1PcCtXfW9VXW6ql4GxoD1SZYDl1TVM9X5P7cfntJncl+PARsnzx4kSQuj3zOE/wT8O+CdrtrVVXUCoL1e1eorgFe72o232oq2PrV+Tp+qOgO8BVzR70FIkuZvxkBI8pvAyar6Tp/77PXJvqapT9dn6li2JRlNMjoxMdHncAbr7MdXQeJynpazH1816H9S6QOrn28q3wz8sySfBT4KXJLkj4DXkyyvqhPtctDJ1n4cWNnVfxg43urDPerdfcaTLAUuBd6YOpCq2g3sBli3bt374lFvS078kL2fOTlzQ/Vl5MmrZm4kaU5mPEOoqnuqariqVtGZLH66qv4FcADY0pptAfa39QPASLtz6Bo6k8eH22WlU0k2tPmBO6f0mdzXbe1nvC/+4EvSB8V8/i+jrwH7kmwFXgFuB6iqo0n2AS8AZ4DtVXW29bkLeAhYBjzRFoAHgEeSjNE5MxiZx7gkSXMwq0Coqm8D327rPwY2vke7HcCOHvVR4IYe9bdpgSJJGgy/qSxJAgwESVJjIEiSAANBktQYCJIkwECQJDUGgiQJMBAkSY2BIEkCDARJUmMgSJIAA0GS1BgIkiTAQJAkNQaCJAkwECRJjYEgSQL6CIQkH01yOMnfJDma5Kut/pUkryV5ti2f7epzT5KxJMeS3NJVvynJkbbtvvZsZdrzlx9t9UNJVl2AY5UkTaOfM4TTwK9W1SeBtcCmJBvatp1VtbYtjwMkWUPnmcjXA5uA+5Msae13AduA1W3Z1OpbgTer6jpgJ3DvvI9MkjQrMwZCdfysvf1wW2qaLpuBvVV1uqpeBsaA9UmWA5dU1TNVVcDDwK1dffa09ceAjZNnD5KkhdHXHEKSJUmeBU4CT1XVobbpi0meS/JgkstabQXwalf38VZb0dan1s/pU1VngLeAK2Z/OJKkueorEKrqbFWtBYbpfNq/gc7ln2vpXEY6AXy9Ne/1yb6mqU/X5xxJtiUZTTI6MTHRz9AlSX2a1V1GVfUT4NvApqp6vQXFO8A3gPWt2TiwsqvbMHC81Yd71M/pk2QpcCnwRo+fv7uq1lXVuqGhodkMXZI0g37uMhpK8rG2vgz4NeB7bU5g0ueA59v6AWCk3Tl0DZ3J48NVdQI4lWRDmx+4E9jf1WdLW78NeLrNM0iSFsjSPtosB/a0O4U+BOyrqm8leSTJWjqXdn4AfAGgqo4m2Qe8AJwBtlfV2bavu4CHgGXAE20BeAB4JMkYnTODkfkfmiRpNmYMhKp6DrixR/3z0/TZAezoUR8FbuhRfxu4faaxSJIuHL+pLEkCDARJUmMgSJIAA0GS1BgIkiTAQJAkNQaCJAkwECRJjYEgSQIMBElSYyBIkgADQZLUGAiSJMBAkCQ1BoIkCTAQJEmNgSBJAvp7pvJHkxxO8jdJjib5aqtfnuSpJC+118u6+tyTZCzJsSS3dNVvSnKkbbuvPVuZ9vzlR1v9UJJVF+BYJUnT6OcM4TTwq1X1SWAtsCnJBuBu4GBVrQYOtvckWUPnmcjXA5uA+9vzmAF2AduA1W3Z1OpbgTer6jpgJ3Dv/A9NkjQbMwZCdfysvf1wWwrYDOxp9T3ArW19M7C3qk5X1cvAGLA+yXLgkqp6pqoKeHhKn8l9PQZsnDx7kCQtjL7mEJIsSfIscBJ4qqoOAVdX1QmA9npVa74CeLWr+3irrWjrU+vn9KmqM8BbwBVzOB5J0hz1FQhVdbaq1gLDdD7t3zBN816f7Gua+nR9zt1xsi3JaJLRiYmJGUYtSZqNWd1lVFU/Ab5N59r/6+0yEO31ZGs2Dqzs6jYMHG/14R71c/okWQpcCrzR4+fvrqp1VbVuaGhoNkOXJM2gn7uMhpJ8rK0vA34N+B5wANjSmm0B9rf1A8BIu3PoGjqTx4fbZaVTSTa0+YE7p/SZ3NdtwNNtnkGStECW9tFmObCn3Sn0IWBfVX0ryTPAviRbgVeA2wGq6miSfcALwBlge1Wdbfu6C3gIWAY80RaAB4BHkozROTMYOR8HJ0nq34yBUFXPATf2qP8Y2PgefXYAO3rUR4F3zT9U1du0QJEkDYbfVJYkAQaCJKkxECRJgIEgSWoMBEkSYCBIkhoDQZIEGAiSpMZAkCQBBoIkqTEQJEmAgSBJagwESRJgIEiSGgNBkgQYCJKkxkCQJAH9PVN5ZZK/SPJikqNJvtTqX0nyWpJn2/LZrj73JBlLcizJLV31m5Icadvua89Wpj1/+dFWP5Rk1QU4VknSNPo5QzgD/E5V/TKwAdieZE3btrOq1rblcYC2bQS4HtgE3N+exwywC9gGrG7LplbfCrxZVdcBO4F7539okqTZmDEQqupEVX23rZ8CXgRWTNNlM7C3qk5X1cvAGLA+yXLgkqp6pqoKeBi4tavPnrb+GLBx8uxBkrQwZjWH0C7l3AgcaqUvJnkuyYNJLmu1FcCrXd3GW21FW59aP6dPVZ0B3gKumM3YJEnz03cgJPkF4E+B366qn9K5/HMtsBY4AXx9smmP7jVNfbo+U8ewLcloktGJiYl+hy5J6kNfgZDkw3TC4I+r6psAVfV6VZ2tqneAbwDrW/NxYGVX92HgeKsP96if0yfJUuBS4I2p46iq3VW1rqrWDQ0N9XeEkqS+9HOXUYAHgBer6ve76su7mn0OeL6tHwBG2p1D19CZPD5cVSeAU0k2tH3eCezv6rOlrd8GPN3mGSRJC2RpH21uBj4PHEnybKv9LnBHkrV0Lu38APgCQFUdTbIPeIHOHUrbq+ps63cX8BCwDHiiLdAJnEeSjNE5MxiZz0FJkmZvxkCoqr+i9zX+x6fpswPY0aM+CtzQo/42cPtMY5EkXTh+U1mSBBgIkqTGQJAkAQaCJKkxECRJgIEgSWoMBEkSYCBIkhoDQZIEGAiSpMZAkCQBBoIkqTEQJEmAgSBJagwESRJgIEiSGgNBkgT090zllUn+IsmLSY4m+VKrX57kqSQvtdfLuvrck2QsybEkt3TVb0pypG27rz1bmfb85Udb/VCSVRfgWCVJ0+jnDOEM8DtV9cvABmB7kjXA3cDBqloNHGzvadtGgOuBTcD9SZa0fe0CtgGr27Kp1bcCb1bVdcBO4N7zcGySpFmYMRCq6kRVfbetnwJeBFYAm4E9rdke4Na2vhnYW1Wnq+plYAxYn2Q5cElVPVNVBTw8pc/kvh4DNk6ePUiSFsas5hDapZwbgUPA1VV1AjqhAVzVmq0AXu3qNt5qK9r61Po5farqDPAWcMVsxiZJmp++AyHJLwB/Cvx2Vf10uqY9ajVNfbo+U8ewLcloktGJiYmZhixJmoW+AiHJh+mEwR9X1Tdb+fV2GYj2erLVx4GVXd2HgeOtPtyjfk6fJEuBS4E3po6jqnZX1bqqWjc0NNTP0CVJfernLqMADwAvVtXvd206AGxp61uA/V31kXbn0DV0Jo8Pt8tKp5JsaPu8c0qfyX3dBjzd5hkkSQtkaR9tbgY+DxxJ8myr/S7wNWBfkq3AK8DtAFV1NMk+4AU6dyhtr6qzrd9dwEPAMuCJtkAncB5JMkbnzGBkfoclSZqtGQOhqv6K3tf4ATa+R58dwI4e9VHghh71t2mBIkkaDL+pLEkCDARJUmMgSJIAA0GS1BgIkiTAQJAkNQaCJAkwECRJjYEgSQIMBElSYyBIkgADQZLUGAiSJMBAkCQ1BoIkCTAQJEmNgSBJAvp7pvKDSU4meb6r9pUkryV5ti2f7dp2T5KxJMeS3NJVvynJkbbtvvZcZdqzlx9t9UNJVp3nY5Qk9aGfM4SHgE096juram1bHgdIsobO85Cvb33uT7Kktd8FbANWt2Vyn1uBN6vqOmAncO8cj0WSNA8zBkJV/SWdB9/3YzOwt6pOV9XLwBiwPsly4JKqeqaqCngYuLWrz562/hiwcfLsQZK0cOYzh/DFJM+1S0qXtdoK4NWuNuOttqKtT62f06eqzgBvAVfMY1ySpDmYayDsAq4F1gIngK+3eq9P9jVNfbo+75JkW5LRJKMTExOzGrAkaXpzCoSqer2qzlbVO8A3gPVt0ziwsqvpMHC81Yd71M/pk2QpcCnvcYmqqnZX1bqqWjc0NDSXoUuS3sOcAqHNCUz6HDB5B9IBYKTdOXQNncnjw1V1AjiVZEObH7gT2N/VZ0tbvw14us0zSJIW0NKZGiT5E+DTwJVJxoEvA59OspbOpZ0fAF8AqKqjSfYBLwBngO1Vdbbt6i46dywtA55oC8ADwCNJxuicGYych+OSJM3SjIFQVXf0KD8wTfsdwI4e9VHghh71t4HbZxqHJOnC8pvKkiTAQJAkNQaCJAkwECRJjYEgSQIMBElSYyBIkgADQZLUGAiSJMBAkCQ1BoIkCTAQJEmNgSBJAgwESVJjIEiSAANBktQYCJIkoI9ASPJgkpNJnu+qXZ7kqSQvtdfLurbdk2QsybEkt3TVb0pypG27rz1bmfb85Udb/VCSVef5GCVJfejnDOEhYNOU2t3AwapaDRxs70myhs4zka9vfe5PsqT12QVsA1a3ZXKfW4E3q+o6YCdw71wPRpI0dzMGQlX9JfDGlPJmYE9b3wPc2lXfW1Wnq+plYAxYn2Q5cElVPVNVBTw8pc/kvh4DNk6ePUiSFs5c5xCurqoTAO31qlZfAbza1W681Va09an1c/pU1RngLeCKOY5LkjRH53tSudcn+5qmPl2fd+882ZZkNMnoxMTEHIcoSeplroHwersMRHs92erjwMqudsPA8VYf7lE/p0+SpcClvPsSFQBVtbuq1lXVuqGhoTkOXZLUy1wD4QCwpa1vAfZ31UfanUPX0Jk8PtwuK51KsqHND9w5pc/kvm4Dnm7zDJKkBbR0pgZJ/gT4NHBlknHgy8DXgH1JtgKvALcDVNXRJPuAF4AzwPaqOtt2dRedO5aWAU+0BeAB4JEkY3TODEbOy5FJkmZlxkCoqjveY9PG92i/A9jRoz4K3NCj/jYtUCRJg+M3lSVJgIEgSWoMBEkSYCBIkhoDQZIEGAiSpMZAkCQBBoIkqTEQJEmAgSBJagwESRJgIEiSGgNBkgQYCJKkxkCQJAEGgiSpMRAkScA8AyHJD5IcSfJsktFWuzzJU0leaq+XdbW/J8lYkmNJbumq39T2M5bkvvbcZUnSAjofZwj/tKrWVtW69v5u4GBVrQYOtvckWUPnecnXA5uA+5MsaX12AduA1W3ZdB7GJUmahQtxyWgzsKet7wFu7arvrarTVfUyMAasT7IcuKSqnqmqAh7u6iPpAjn78VWQuJyn5ezHVw36n3Tels6zfwFPJingD6pqN3B1VZ0AqKoTSa5qbVcA/6ur73ir/X1bn1qXdAEtOfFD9n7m5KCH8YEx8uRVMze6yM03EG6uquPtj/5TSb43Tdte8wI1Tf3dO0i20bm0xCc+8YnZjlWSNI15XTKqquPt9STwZ8B64PV2GYj2OvkRZBxY2dV9GDje6sM96r1+3u6qWldV64aGhuYzdEnSFHMOhCQ/n+QXJ9eBzwDPAweALa3ZFmB/Wz8AjCT5SJJr6EweH26Xl04l2dDuLrqzq48kaYHM55LR1cCftTtElwL/tar+PMlfA/uSbAVeAW4HqKqjSfYBLwBngO1Vdbbt6y7gIWAZ8ERbJEkLaM6BUFXfBz7Zo/5jYON79NkB7OhRHwVumOtYJEnz5zeVJUmAgSBJagwESRJgIEiSGgNBkgQYCJKkxkCQJAEGgiSpMRAkSYCBIElqDARJEmAgSJIaA0GSBBgIkqTGQJAkAQaCJKkxECRJwEUUCEk2JTmWZCzJ3YMejyQtNhdFICRZAvxn4NeBNcAdSdYMdlSStLhcFIEArAfGqur7VfV3wF5g84DHJEmLysUSCCuAV7vej7eaJGmBLB30AJr0qNW7GiXbgG3t7c+SHLugozpfnrxq0CPox5XAjwY9iJncAZBevy6aE383z5v30e/mP3yvDRdLIIwDK7veDwPHpzaqqt3A7oUa1GKSZLSq1g16HNJU/m4unIvlktFfA6uTXJPk54AR4MCAxyRJi8pFcYZQVWeSfBH478AS4MGqOjrgYUnSonJRBAJAVT0OPD7ocSxiXorTxcrfzQWSqnfN3UqSFqGLZQ5BkjRgBoIkCTAQJEmNgbCIJbk0yc4ko235epJLBz0uLW5Jbk/yi2393yf5ZpJ/MuhxLQYGwuL2IPBT4J+35afAfxnoiCT4D1V1KsmngFuAPcCuAY9pUTAQFrdrq+rL7T8V/H5VfRX4R4MelBa9s+31N4BdVbUf+LkBjmfRMBAWt79tn8IASHIz8LcDHI8E8FqSP6Bz1vp4ko/g36oF4fcQFrEka+mcjk/OG7wJbKmq5wY2KC16Sf4BsAk4UlUvJVkO/EpVPTngoX3gXTTfVNZAvAj8HnAt8DHgLeBWwEDQwFTV/0tyEvgU8BJwpr3qAjMQFrf9wE+A7wKvDXYoUkeSLwPrgF+ic5PDh4E/Am4e5LgWAwNhcRuuqk2DHoQ0xeeAG+l8UKGqjk/ehqoLy4maxe1/JvmVQQ9CmuLvqjO5WQBJfn7A41k0PENY3D4F/MskLwOn6Ty5rqrqHw92WFrk9rW7jD6W5N8A/xr4xoDHtCgYCIvbrw96AFIPp4H/QeeLkr8E/MeqemqwQ1ocDIRFrKp+OOgxSD1cDXyJzhzCg3TCQQvA7yFIuugkCfAZ4F/RueNoH/BAVf2fgQ7sA85JZUkXnTap/H/bcga4DHgsye8NdGAfcJ4hSLqoJPm3wBbgR8AfAv+tqv4+yYeAl6rq2oEO8APMOQRJF5srgd+aOsdVVe8k+c0BjWlR8AxBkgQ4hyBJagwESRJgIEiSGgNBkgQYCJKk5v8DvCx2fRC75LYAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Bar plot of freequencies\n",
    "data['y'].value_counts().plot.bar(color = np.random.rand(3,), ec='red')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29f696ea",
   "metadata": {},
   "source": [
    "Here, We can see that 5289 users out of total 39922 have subscribed which is around 12%."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "24355535",
   "metadata": {},
   "source": [
    "Let's now explore the variables to have a better understanding of the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c804614e",
   "metadata": {},
   "source": [
    "Let's first look at the distribution of age variable to see how many people belongs to a particular age group."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "2c5e304b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x1c255c78e50>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sn.displot(data[\"age\"], color=np.random.rand(3,))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3046cbd0",
   "metadata": {},
   "source": [
    "Now, We can infer that most of the clients fall in the age group between 20-60."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "517b853e",
   "metadata": {},
   "source": [
    "Now let's look at what are the different types of jobs of the clients. As job is a categorical variable, we will look at its frequency table."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "df152f52",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYQAAAE1CAYAAADnK5cDAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAoT0lEQVR4nO3df9zl9Zz/8ceziX7IqDS1qZi0iYpWDSKbkh9F1CKyok1rbFrF+vIN+xXt5seuFrFF+jVhJT+LjSIqkTKp9EtbW9QoNUhGKOX5/eP9Ps2Za87M1HV93p9r5rqe99vtul3nfM45n9fn+nHO6/N5/3i9ZZuIiIjVJvsAIiJi5ZCEEBERQBJCRERUSQgREQEkIURERJWEEBERAKw+2QcwXhtssIFnz5492YcREbFKueSSS35pe9aox1aYECSdCOwJ3G5727ptfeBzwGzgp8DLbd9RH3s7cCBwH3CI7bPq9h2Ak4G1gDOBQ21b0hrAKcAOwK+AV9j+6YqOa/bs2cyfP39FT4uIiCGSfrasxx5Ik9HJwO5jth0GnGN7S+Cceh9JWwP7AtvU1xwjaUZ9zbHAXGDL+jXY54HAHbb/EvgQ8IEHcEwREdGxFSYE2+cDvx6zeS9gXr09D9h7aPuptu+2fSNwPfBUSRsDM21f6DI1+pQxrxns6wvAbpI0vh8nIiLGa7ydyhvZvhWgft+wbt8EuHnoeQvqtk3q7bHbl3iN7XuBO4FHjvO4IiJinLoeZTTqzN7L2b681yy9c2mupPmS5i9cuHCchxgREaOMNyHcVpuBqN9vr9sXAJsNPW9T4Ja6fdMR25d4jaTVgUewdBMVALaPsz3H9pxZs0Z2kkdExDiNNyGcAexfb+8PnD60fV9Ja0janNJ5fHFtVlokacfaP/CaMa8Z7OtlwLedEqwREb17IMNOPwvsAmwgaQFwOPB+4DRJBwI3AfsA2L5K0mnA1cC9wMG276u7OojFw06/Xr8ATgA+Jel6ypXBvp38ZBER8aBoVT0ZnzNnjjMPISLiwZF0ie05ox5bZWcqL8/l5x4xrtdtt8u7Oj6SiIhVR2oZRUQEkIQQERFVEkJERABJCBERUSUhREQEMEVHGfUto5oiYirIFUJERABJCBERUSUhREQEkIQQERFVEkJERABJCBERUSUhREQEkIQQERFVEkJERABJCBERUSUhREQEkIQQERFVEkJERABJCBERUSUhREQEkIQQERFVEkJERABJCBERUSUhREQEkIQQERFVEkJERABJCBERUSUhREQEkIQQERFVEkJERABJCBERUU0oIUh6s6SrJF0p6bOS1pS0vqRvSrqufl9v6Plvl3S9pGslPX9o+w6SrqiPHS1JEzmuiIh48MadECRtAhwCzLG9LTAD2Bc4DDjH9pbAOfU+krauj28D7A4cI2lG3d2xwFxgy/q1+3iPKyIixmeiTUarA2tJWh1YG7gF2AuYVx+fB+xdb+8FnGr7bts3AtcDT5W0MTDT9oW2DZwy9JqIiOjJuBOC7Z8DHwRuAm4F7rR9NrCR7Vvrc24FNqwv2QS4eWgXC+q2TertsdsjIqJHE2kyWo9y1r858CjgYZL2W95LRmzzcraPijlX0nxJ8xcuXPhgDzkiIpZjIk1GzwFutL3Q9p+ALwHPAG6rzUDU77fX5y8ANht6/aaUJqYF9fbY7UuxfZztObbnzJo1awKHHhERY00kIdwE7Chp7ToqaDfgGuAMYP/6nP2B0+vtM4B9Ja0haXNK5/HFtVlpkaQd635eM/SaiIjoyerjfaHtiyR9AfgRcC9wKXAcsA5wmqQDKUljn/r8qySdBlxdn3+w7fvq7g4CTgbWAr5evyIiokfjTggAtg8HDh+z+W7K1cKo5x8JHDli+3xg24kcS0RETExmKkdEBJCEEBERVRJCREQASQgREVFNqFM5Jsfl5x4xrtdtt8u7Oj6SiJhKcoUQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERHABBOCpHUlfUHSTyRdI+npktaX9E1J19Xv6w09/+2Srpd0raTnD23fQdIV9bGjJWkixxUREQ/eRK8QPgJ8w/bjge2Aa4DDgHNsbwmcU+8jaWtgX2AbYHfgGEkz6n6OBeYCW9av3Sd4XBER8SCNOyFImgnsDJwAYPse278B9gLm1afNA/aut/cCTrV9t+0bgeuBp0raGJhp+0LbBk4Zek1ERPRkIlcIjwUWAidJulTS8ZIeBmxk+1aA+n3D+vxNgJuHXr+gbtuk3h67PSIiejSRhLA6sD1wrO0nA3dRm4eWYVS/gJezfekdSHMlzZc0f+HChQ/2eCMiYjkmkhAWAAtsX1Tvf4GSIG6rzUDU77cPPX+zoddvCtxSt286YvtSbB9ne47tObNmzZrAoUdExFjjTgi2fwHcLGmrumk34GrgDGD/um1/4PR6+wxgX0lrSNqc0nl8cW1WWiRpxzq66DVDr4mIiJ6sPsHXvxH4jKSHAjcAB1CSzGmSDgRuAvYBsH2VpNMoSeNe4GDb99X9HAScDKwFfL1+RUREjyaUEGxfBswZ8dBuy3j+kcCRI7bPB7adyLFERMTEZKZyREQASQgREVElIUREBJCEEBERVRJCREQASQgREVElIUREBJCEEBERVRJCREQASQgREVElIUREBJCEEBERVRJCREQAEy9/HdPA5eceMa7XbbfLuzo+kohoKVcIEREBJCFERESVhBAREUASQkREVEkIEREBJCFERESVhBAREUASQkREVEkIEREBJCFERESVhBAREUASQkREVEkIEREBJCFERESVhBAREUASQkREVEkIEREBJCFERESVhBAREUASQkREVKtPdAeSZgDzgZ/b3lPS+sDngNnAT4GX276jPvftwIHAfcAhts+q23cATgbWAs4EDrXtiR5brJouP/eIcb1uu13e1fGRREwvXVwhHApcM3T/MOAc21sC59T7SNoa2BfYBtgdOKYmE4BjgbnAlvVr9w6OKyIiHoQJJQRJmwIvBI4f2rwXMK/engfsPbT9VNt3274RuB54qqSNgZm2L6xXBacMvSYiInoy0SuEDwNvA/48tG0j27cC1O8b1u2bADcPPW9B3bZJvT12+1IkzZU0X9L8hQsXTvDQIyJi2LgTgqQ9gdttX/JAXzJim5ezfemN9nG259ieM2vWrAcYNiIiHoiJdCrvBLxY0guANYGZkj4N3CZpY9u31uag2+vzFwCbDb1+U+CWun3TEdsjIqJH475CsP1225vank3pLP627f2AM4D969P2B06vt88A9pW0hqTNKZ3HF9dmpUWSdpQk4DVDr4mIiJ5MeNjpCO8HTpN0IHATsA+A7asknQZcDdwLHGz7vvqag1g87PTr9SsiInrUSUKwfS5wbr39K2C3ZTzvSODIEdvnA9t2cSwRETE+makcERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERASQhRERElYQQERFAEkJERFRJCBERAcDqk30AEZPt8nOPGNfrttvlXR0fScTkyhVCREQAE0gIkjaT9B1J10i6StKhdfv6kr4p6br6fb2h17xd0vWSrpX0/KHtO0i6oj52tCRN7MeKiIgHayJXCPcCb7H9BGBH4GBJWwOHAefY3hI4p96nPrYvsA2wO3CMpBl1X8cCc4Et69fuEziuiIgYh3EnBNu32v5Rvb0IuAbYBNgLmFefNg/Yu97eCzjV9t22bwSuB54qaWNgpu0LbRs4Zeg1ERHRk076ECTNBp4MXARsZPtWKEkD2LA+bRPg5qGXLajbNqm3x26PiIgeTTghSFoH+CLwJtu/Xd5TR2zzcraPijVX0nxJ8xcuXPjgDzYiIpZpQglB0kMoyeAztr9UN99Wm4Go32+v2xcAmw29fFPglrp90xHbl2L7ONtzbM+ZNWvWRA49IiLGmMgoIwEnANfY/o+hh84A9q+39wdOH9q+r6Q1JG1O6Ty+uDYrLZK0Y93na4ZeExERPZnIxLSdgFcDV0i6rG57B/B+4DRJBwI3AfsA2L5K0mnA1ZQRSgfbvq++7iDgZGAt4Ov1KyIiejTuhGD7Aka3/wPstozXHAkcOWL7fGDb8R5LRERMXGYqR0QEkIQQERFVEkJERABJCBERUaX8dUSPUmo7Vma5QoiICCBXCBFTWq5I4sHIFUJERABJCBERUSUhREQEkD6EiOhQ+ixWbUkIEbHKSgLqVpqMIiICSEKIiIgqCSEiIoAkhIiIqJIQIiICSEKIiIgqw04jIh6gqT7MNVcIEREBJCFERESVhBAREUASQkREVEkIEREBJCFERESVhBAREUASQkREVEkIEREBJCFERESVhBAREUASQkREVCluFxGxkuq7mF6uECIiAkhCiIiIaqVJCJJ2l3StpOslHTbZxxMRMd2sFAlB0gzgP4E9gK2BV0raenKPKiJielkpEgLwVOB62zfYvgc4Fdhrko8pImJaWVkSwibAzUP3F9RtERHRE9me7GNA0j7A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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['job'].value_counts().plot.bar(color = np.random.rand(3,))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f85a3df7",
   "metadata": {},
   "source": [
    "We see that most of the clients belongs to blue-collar job and the students are least in number as students generally do not take a term deposit."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f745a46",
   "metadata": {},
   "source": [
    "Now, let's look at the education background of the customes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "92d0ec35",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYQAAAEiCAYAAAD5+KUgAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAVfElEQVR4nO3dfdSkdX3f8fdHNiEYBXlYCC7UJbqSrCSgrIjRtiolLPUBTMEuJwZOJF2l2NS2SQvpafTYQ5HWhhM8kWQNymKIuJpY8DSIGyAhGgIuFuVJwiqoKxxYxCBNCnbx2z+u353O3szu3jv3zH3ds7xf58yZme9c1+x3rrO7n/n9rodJVSFJ0nP6bkCStDgYCJIkwECQJDUGgiQJMBAkSY2BIEkCYEnfDYzqoIMOquXLl/fdhiRNldtuu+3Rqlo67LWpDYTly5ezadOmvtuQpKmS5Js7es0pI0kSYCBIkhoDQZIEGAiSpMZAkCQBBoIkqTEQJEmAgSBJaqb2xLRJeOz8E/tuYU4OuHBj3y1I2gM5QpAkAQaCJKkxECRJgIEgSWoMBEkSYCBIkhoDQZIEGAiSpMZAkCQBBoIkqTEQJEmAgSBJagwESRJgIEiSGgNBkgQYCJKkxkCQJAEGgiSpMRAkSYCBIElqDARJEmAgSJIaA0GSBBgIkqRml4GQ5PAkNya5J8ldSf51qx+QZGOS+9r9/gPrnJ9kc5J7k5w0UD82yR3ttUuSpNX3TvLJVr8lyfIJfFZJ0k7MZYSwDfh3VfXTwPHAuUlWAucB11fVCuD69pz22hrgZcBq4MNJ9mrvdSmwFljRbqtb/Wzge1X1EuBi4KIxfDZJ0m7YZSBU1UNV9eX2+AngHmAZcAqwvi22Hji1PT4FuKqqnqqq+4HNwHFJDgX2raqbq6qAK2atM/NenwZOmBk9SJIWxm7tQ2hTOS8HbgEOqaqHoAsN4OC22DLg2wOrbWm1Ze3x7Pp261TVNuBx4MDd6U2SND9zDoQkzwP+CHhPVX1/Z4sOqdVO6jtbZ3YPa5NsSrJp69atu2pZkrQb5hQISX6ELgyurKo/buWH2zQQ7f6RVt8CHD6w+mHAg61+2JD6duskWQLsBzw2u4+qWldVq6pq1dKlS+fSuiRpjuZylFGAy4B7quq3Bl66BjirPT4LuHqgvqYdOXQE3c7jW9u00hNJjm/veeasdWbe6zTghrafQZK0QJbMYZnXAL8E3JHk9lb7DeADwIYkZwPfAk4HqKq7kmwA7qY7Quncqnq6rXcOcDmwD3Btu0EXOB9PspluZLBmfh9LkrS7dhkIVfUFhs/xA5ywg3UuAC4YUt8EHDWk/iQtUCRJ/fBMZUkSYCBIkhoDQZIEGAiSpMZAkCQBBoIkqTEQJEmAgSBJagwESRJgIEiSGgNBkgQYCJKkxkCQJAEGgiSpMRAkSYCBIElqDARJEmAgSJIaA0GSBBgIkqTGQJAkAQaCJKkxECRJgIEgSWoMBEkSYCBIkhoDQZIEGAiSpMZAkCQBBoIkqTEQJEmAgSBJagwESRJgIEiSGgNBkgQYCJKkZpeBkOSjSR5JcudA7X1JvpPk9nb7pwOvnZ9kc5J7k5w0UD82yR3ttUuSpNX3TvLJVr8lyfIxf0ZJ0hzMZYRwObB6SP3iqjqm3f4EIMlKYA3wsrbOh5Ps1Za/FFgLrGi3mfc8G/heVb0EuBi4aMTPIkmah10GQlXdBDw2x/c7Bbiqqp6qqvuBzcBxSQ4F9q2qm6uqgCuAUwfWWd8efxo4YWb0IElaOPPZh/DuJF9tU0r7t9oy4NsDy2xptWXt8ez6dutU1TbgceDAYX9gkrVJNiXZtHXr1nm0LkmabdRAuBR4MXAM8BDw31t92Df72kl9Z+s8s1i1rqpWVdWqpUuX7lbDkqSdGykQqurhqnq6qn4IfAQ4rr20BTh8YNHDgAdb/bAh9e3WSbIE2I+5T1FJksZkpEBo+wRmvBWYOQLpGmBNO3LoCLqdx7dW1UPAE0mOb/sHzgSuHljnrPb4NOCGtp9BkrSAluxqgSSfAF4HHJRkC/Be4HVJjqGb2nkAeCdAVd2VZANwN7ANOLeqnm5vdQ7dEUv7ANe2G8BlwMeTbKYbGawZw+eSJO2mXQZCVZ0xpHzZTpa/ALhgSH0TcNSQ+pPA6bvqQ5I0WZ6pLEkCDARJUmMgSJIAA0GS1BgIkiTAQJAkNQaCJAkwECRJjYEgSQLmcKayNKrHzj+x7xbm5IALN/bdgrQoOEKQJAEGgiSpMRAkSYCBIElqDARJEmAgSJIaA0GSBBgIkqTGQJAkAQaCJKkxECRJgIEgSWoMBEkSYCBIkhoDQZIEGAiSpMZAkCQBBoIkqTEQJEmAgSBJagwESRJgIEiSGgNBkgQYCJKkxkCQJAFzCIQkH03ySJI7B2oHJNmY5L52v//Aa+cn2Zzk3iQnDdSPTXJHe+2SJGn1vZN8stVvSbJ8zJ9RkjQHcxkhXA6snlU7D7i+qlYA17fnJFkJrAFe1tb5cJK92jqXAmuBFe02855nA9+rqpcAFwMXjfphJEmj22UgVNVNwGOzyqcA69vj9cCpA/Wrquqpqrof2Awcl+RQYN+qurmqCrhi1joz7/Vp4ISZ0YMkaeGMug/hkKp6CKDdH9zqy4BvDyy3pdWWtcez69utU1XbgMeBA0fsS5I0onHvVB72zb52Ut/ZOs9882Rtkk1JNm3dunXEFiVJw4waCA+3aSDa/SOtvgU4fGC5w4AHW/2wIfXt1kmyBNiPZ05RAVBV66pqVVWtWrp06YitS5KGGTUQrgHOao/PAq4eqK9pRw4dQbfz+NY2rfREkuPb/oEzZ60z816nATe0/QySpAW0ZFcLJPkE8DrgoCRbgPcCHwA2JDkb+BZwOkBV3ZVkA3A3sA04t6qebm91Dt0RS/sA17YbwGXAx5NsphsZrBnLJ5Mk7ZZdBkJVnbGDl07YwfIXABcMqW8CjhpSf5IWKJKk/nimsiQJMBAkSY2BIEkCDARJUmMgSJIAA0GS1BgIkiTAQJAkNQaCJAkwECRJjYEgSQIMBElSYyBIkgADQZLUGAiSJMBAkCQ1BoIkCTAQJEmNgSBJAgwESVJjIEiSAANBktQYCJIkAJb03YCkXXvs/BP7bmFODrhwY98taB4cIUiSAANBktQYCJIkwECQJDUGgiQJMBAkSY2BIEkCDARJUmMgSJIAA0GS1BgIkiTAQJAkNfMKhCQPJLkjye1JNrXaAUk2Jrmv3e8/sPz5STYnuTfJSQP1Y9v7bE5ySZLMpy9J0u4bxwjh9VV1TFWtas/PA66vqhXA9e05SVYCa4CXAauBDyfZq61zKbAWWNFuq8fQlyRpN0xiyugUYH17vB44daB+VVU9VVX3A5uB45IcCuxbVTdXVQFXDKwjSVog8w2EAj6f5LYka1vtkKp6CKDdH9zqy4BvD6y7pdWWtcez68+QZG2STUk2bd26dZ6tS5IGzfcHcl5TVQ8mORjYmORrO1l22H6B2kn9mcWqdcA6gFWrVg1dRpI0mnmNEKrqwXb/CPAZ4Djg4TYNRLt/pC2+BTh8YPXDgAdb/bAhdUnSAho5EJL8eJLnzzwGfh64E7gGOKstdhZwdXt8DbAmyd5JjqDbeXxrm1Z6Isnx7eiiMwfWkSQtkPlMGR0CfKYdIboE+MOq+lySLwEbkpwNfAs4HaCq7kqyAbgb2AacW1VPt/c6B7gc2Ae4tt0kSQto5ECoqm8ARw+pfxc4YQfrXABcMKS+CThq1F4kSfPnmcqSJMBAkCQ1BoIkCTAQJEmNgSBJAgwESVJjIEiSAANBktQYCJIkwECQJDUGgiQJMBAkSY2BIEkCDARJUmMgSJIAA0GS1BgIkiTAQJAkNQaCJAkwECRJjYEgSQIMBElSs6TvBiRpoT12/ol9tzAnB1y4cUH/PEcIkiTAQJAkNQaCJAkwECRJjYEgSQIMBElSYyBIkgADQZLUGAiSJMBAkCQ1BoIkCTAQJEmNgSBJAhZRICRZneTeJJuTnNd3P5L0bLMoAiHJXsDvACcDK4EzkqzstytJenZZFIEAHAdsrqpvVNUPgKuAU3ruSZKeVVJVffdAktOA1VX1K+35LwGvqqp3z1puLbC2PT0SuHdBGx3NQcCjfTexB3F7jo/bcrymZXu+qKqWDnthsfxiWobUnpFUVbUOWDf5dsYnyaaqWtV3H3sKt+f4uC3Ha0/YnotlymgLcPjA88OAB3vqRZKelRZLIHwJWJHkiCQ/CqwBrum5J0l6VlkUU0ZVtS3Ju4HrgL2Aj1bVXT23NS5TNcU1Bdye4+O2HK+p356LYqeyJKl/i2XKSJLUMwNBkgQYCJKkxkCYgCRvSuK2lTRV3Kk8AUn+AHg18EfAx6rqnp5bmlpJPki3DfeUo8565fYcryR7A/8MWM7AUZtV9f6+epoPv8VOQFW9HXg58HXgY0luTrI2yfN7bm0afQ1Yl+SWJO9Ksl/fDU05t+d4XU133bVtwN8O3KaSI4QJSnIQ8HbgPcA9wEuAS6rqQ332NY2SHAn8MnAG8EXgI1V1Y79dTS+353gkubOqjuq7j3FxhDABSd6S5DPADcCPAMdV1cnA0cCv9drcFGqXR/+pdnsU+Arwb5Nc1WtjU8rtOVZ/meRn+m5iXBwhTECS9cBlVXXTkNdOqKrre2hrKiX5LeDNdOF6WVXdOvDavVV1ZG/NTSG353gluZtu5H8/8BTdhTqrqn6218ZGtCguXbEnad++lg0LAwDDYO6SBPgecHRV/d2QRY5b4JammttzIk7uu4FxcspozKrqaeDv3Fk3f9UNX0/dwX9eVNXjC9zSVHN7TsTZwEuBR6vqmzO3vpsalSOEyXgSuCPJRgaOOKiqX+2vpan1V0leWVVf6ruRPYTbc7weoNsxf0mSJ4C/AG6qqqt77WpE7kOYgCRnDatX1fqF7mXatTnalwLfpAvXqZ6j7ZvbczKS/ATwNrqDRvavqqk8xNxA0KKW5EXD6tM8LO+T23O8kvw+sBJ4mG508AXgy1W1rdfGRuSU0QQkWQFcSPcX5cdm6lX1k701NaVm/qNKcjAD21KjcXuO3YF0v+HyN8BjdPsSpjIMwJ3Kk/Ix4FK6sxdfD1wBfLzXjqZUO6fjPrrD+v6cbs722l6bmmJuz/GqqrdW1auA/wq8ALgxyZZ+uxqdgTAZ+7TDS9OOOngf8Iaee5pW/xk4HvjrqjoCOIHuzFqNxu05Ru1ClhcBHwXeRXd+x2/229XonDKajCfb1U7vaz8N+h3g4J57mlb/t6q+m+Q5SZ5TVTe2f4AajdtzvE4GbgJ+u6oe7LuZ+TIQJuM9wHOBX6X7RvYGYOiRR9qlv0nyPLp/dFcmeYRuKk6jcXuOUVWdm+QQ4JVJXgHcWlWP9N3XqDzKSItakh+nO68jwC8C+wFXVtV3e21sSrk9xyvJ6cAHgT+j26b/EPj1qvp0n32NykAYoySfBXa4QavqLQvYjrRDSfZl++v3P9ZjO1MryVeAE2dGBUmWAn9aVUf329lonDIarw+2+18AfgL4g/b8DLqjOTRHSb5QVa9tZ38OhuzMiVT79tTaVEvyTuD9wP8BfkjbnoCHRI/mObOmiL7LFB+s4whhApLcVFX/aFc1aaG1Q05fXVWP9t3LniDJfwN+FvhEK/1z4KtV9R/662p0U5tki9zSJH//jSvJEcDSHvuZWkmecf7GsJrm7OvA0IvbafdV1a8D6+hC4Whg3bSGAThCmIgkq+n+knyjlZYD76yq63prakol+XJVvWLg+RK6b2Are2xraiV5Od2Jk7fQXb8f8MKL6rgPYQKq6nPt8hU/1Upfq6qndraOtpfkfOA3gH2SfH+mDPyALmw1mt+jO3nqDrp9CJqHJL8AXER3nlGY8n1cjhAmJMnP0Y0MBo/kuKK3hqZQO7nv96vqHX33sqdI8pdV9XN997GnSLIZeHNV3dN3L+PgCGEC2hz3i4HbgadbueiuaaQ5qqofJpnKw/cWsRuTrAU+y/ZTRh52OpqH95QwAEcIE5HkHmBluXHnLcnvAJf7gy7jkeT+IeXySryjSfLbdIeY/w+2D9g/7qun+XCEMBl30v0leajvRvYArwfeleQB/EGXeWsXtNP47Et31NbPD9QKmMpAcIQwAUluBI4BbmX7bw2eqbyb/EGX8Ujyhqq6oe0EfYZp/UbbtyQHzJ5uS3JEVQ0biS16jhAm4319N7CnqKpvJnktsKKqPtYuDfC8vvuaQv+Y7uiiNw95bWq/0S4Cn01yclV9HyDJTwOfAo7qt63ROEKYkJkrILanU30FxD4leS+wCjiyql6a5IXAp6rqNT23NnXaUVunVdWGvnvZUyR5I/DvgTcCR9IdOPKLVXV7n32NyjOVJyDJ2+imi06n++HtW5Kc1m9XU+utwFvo9h/Qrjk/lT9g3req+iHw7r772JNU1f8ELgY+D1wOnDqtYQBOGU3KfwReOfsKiMBUXhK3Zz+oqkpS8PeXb9boNib5NeCTtJAFDzvdXUk+xPYXXdyX7soE/yrJ1J75bSBMxh51BcSebUjye8ALkvwL4B3AR3ruaZq9g+4/sn85q+5hp7tn06znt/XSxZgZCJPxuSTXsf0VEP0h89EspRtZfZ9ujvY3gX/Sa0fTbSVdGLyWLhj+AvjdXjuaQlW1vu8eJsGdyhPSDu97Ld1x8zdV1Wd6bmkqzb64Xat91fMQRpNkA124XtlKZwAvqKq39dfV9EryGrqjCl9E9wV75jyZqRxxGQgT0C53/VBVPdme7wMcUlUP9NrYFElyDt032Z+ku2TzjOcDX6yqt/fS2JRL8pXZv+Y1rKa5SfI14N/QTRnNXKaGaf1JUqeMJuNTwOAFxJ5utVcOX1xD/CHdNNuFwHkD9SfcATov/yvJ8VX1VwBJXgV8seeeptnjVbXHTAc7QpiAJLdX1TGzan4LU+/adbaOBL7VSv8AuIfuUtheEmQ3JfkAsBfdiX2DVyX4cm9NzYMjhMnYmuQtVXUNQJJTAH+yUIvB6r4b2MO8qt0f2+5nfqP6Df20Mz8GwmS8C7iyXamzgC3Amf22JHkNqAn4syG1qZ12MRAmoKq+Dhyf5Hl003JP9N2TpIn43wOPfwx4E90U3FRyH8IEtOsY/RfghVV1cpKVwKur6rKeW5M0QUn2Bq6pqpP67mUUnj07GZcD1wEvbM//GnhPX81IWjDPZYrP+nbKaDIOqqoN7YfiqaptSZ7e1UqSpkuSO/j/+wz2ojuz/v39dTQ/BsJk/G2SA2l/UZIcDzzeb0uSJuBNA4+30f3G8ra+mpkv9yFMQJJXAB+i+5GMO+m+NZxWVV/ttTFJ2gn3IUzGi4GT6c5Wvg64D0djkhY5A2Ey/lP7Sb396a7MuQ64tN+WJGnnDITJmNmB/Ebgd6vqauBHe+xHknbJQJiM77QfdXkb8Cft2GS3taRFzZ3KE5DkuXTXjLmjqu5LcijwM1X1+Z5bk6QdMhAkSYDTGJKkxkCQJAEGgiSpMRAkSYCBIElq/h/3i5H09sOBowAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['education_qual'].value_counts().plot.bar(color = np.random.rand(3,))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33dc76b6",
   "metadata": {},
   "source": [
    "We can see that most of our clients have secondary or tertiary level of education."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "95a5d50c",
   "metadata": {},
   "source": [
    "### "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68942bf3",
   "metadata": {},
   "source": [
    "### Bivariate Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "3fb364a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y                no   yes\n",
      "job                      \n",
      "admin.         4540   631\n",
      "blue-collar    9024   708\n",
      "entrepreneur   1364   123\n",
      "housemaid      1131   109\n",
      "management     8157  1301\n",
      "retired        1748   516\n",
      "self-employed  1392   187\n",
      "services       3785   369\n",
      "student         669   269\n",
      "technician     6757   840\n",
      "unemployed     1101   202\n",
      "unknown         254    34\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Percentage')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(pd.crosstab(data['job'],data['y']))\n",
    "\n",
    "job=pd.crosstab(data['job'],data['y'])\n",
    "job.div(job.sum(1).astype(float), axis=0).plot(kind=\"bar\", stacked=True, figsize=(8,8))\n",
    "plt.xlabel('Job')\n",
    "plt.ylabel('Percentage')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c167debb",
   "metadata": {},
   "source": [
    "From the above graph we can infer that students and retired people have higher chances of subscribing to a term deposit, which is surprising as students generally do not subscribe to a term deposit. The possible reason is that the number of students in the dataset is less and comparatively to other job types, more students have subscribed to a term deposit."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b1df654",
   "metadata": {},
   "source": [
    "Next, let's explore the 'education_qual' variable against the 'y' variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8cc23000",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y                  no   yes\n",
      "education_qual             \n",
      "primary          6260   591\n",
      "secondary       20752  2450\n",
      "tertiary        11305  1996\n",
      "unknown          1605   252\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Percentage')"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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O5cFCxZ0HUNqgiUuNW7lypVsvvnmLFy4kCRdx9kgVcWtt97KypUru44iNcHD6VLj7r77bubNm9d8gQMkYd68eTPmqII0bJa4NAPMhAJfayb9LdKwWeKSJDXKEpckqVGWuCRJjbLEJU3o+OOP5/3vf/8Dy+94xzv4wAc+0GEiSWtZ4pImdPjhh3PaaacBsGbNGk4//XQOOeSQjlNJAu8TlzSJhQsXMm/ePC699FJ+/OMfs/vuuzNv3ryuY0nCEpc0gCOOOIKlS5fyox/9iMMOO6zrOJL6PJwuaVIHHXQQZ599NhdddBH77bdf13Ek9TkSlzSpTTbZhL333pstt9ySOXPmdB1HUp8lLmlSa9as4YILLuDMM8/sOoqkMTycLmlC11xzDTvuuCP77LMPixYt6jqOpDEciUua0OLFi7nxxhu7jiFpHI7EJUlqlCUuSVKjLHFJkhpliUuS1CgvbJNmmIVv/9KU7u+mv3rRlO5P0tRxJC5JUqMscUkb7KabbuLJT34yb3jDG9hpp53Yd999+cUvfsFll13GXnvtxS677MJBBx3Ebbfd1nVUaUaxxCVNiRtuuIGjjz6aq6++mi233JLPfe5zvO51r+Pd7343V1xxBTvvvDMnnXRS1zGlGcUSlzQltttuO3bbbTcA9txzT7797W9z++2385znPAeA17/+9Zx33nkdJpRmHktc0pTYdNNNH/h+zpw53H777d2FkWYJS1zSUGyxxRY86lGP4vzzzwfgk5/85AOjcklTw1vMpBlmlG4JO+200zjqqKO466672H777Tn11FO7jiTNKJa4pA22cOFCrrrqqgeWjz322Ae+v+CCC7qIJM0KHk6XJKlRlrgkSY2yxCVJapQlLklSoyxxSZIaZYlLktQobzGTZpoTt5ji/d0xtfuTNGUciUuS1ChH4pI2yPHHH89WW23FMcccA8A73vEOtt56a+655x7OOOMM7rnnHg466CBOOukk7rzzTl75yleycuVK7r//fo4//nhe9apXdfwXSO1yJC5pgxx++OGcdtppAKxZs4bTTz+drbfemhtuuIELL7yQyy67jIsvvpjzzjuPs88+m8c97nFcfvnlXHXVVey///4dp5faZolL2iALFy5k3rx5XHrppZx77rnsvvvuXHTRRQ98v8cee3Dddddxww03sPPOO/OVr3yFt73tbZx//vlsscUU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\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(pd.crosstab(data['education_qual'],data['y']))\n",
    "\n",
    "education_qual=pd.crosstab(data['education_qual'],data['y'])\n",
    "education_qual.div(education_qual.sum(1).astype(float), axis=0).plot(kind=\"bar\", stacked=True, figsize=(8,8))\n",
    "plt.xlabel('education_qual')\n",
    "plt.ylabel('Percentage')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d59d7171",
   "metadata": {},
   "source": [
    "We can infer that clients having education_qual : tertiary and unknown have slightly higher chances of subscribing to a insurance policy as compared to the other clients."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7de0bf25",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y            no   yes\n",
      "marital              \n",
      "divorced   4585   622\n",
      "married   24459  2755\n",
      "single    10878  1912\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Percentage')"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfEAAAIBCAYAAABZSIb5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAfhklEQVR4nO3de7SddX3n8ffHQEQnkEiIjtyaCGlrHBAwIrQdFbFctJamdkDEogIiI1Q6Ux2kFJTam+20HW0tDMtiUm8Uqla0KaBWhdZJDRHkKiVFlIAoIKAFuYR854+9Q08PJyc755wnm9/m/VrrrJzn2c/Z+8tah7zzXPazU1VIkqT2PG3YA0iSpKkx4pIkNcqIS5LUKCMuSVKjjLgkSY0y4pIkNWqbYQ+wpXbaaadauHDhsMeQJGmrWbNmzd1VtWD8+uYivnDhQq688sphjyFJ0laT5NsTrfdwuiRJjTLikiQ1yohLktSo5s6JS5IE8Oijj7Ju3ToeeuihYY8yY7bbbjt23XVXtt1224G2N+KSpCatW7eO7bffnoULF5Jk2ONMW1Vxzz33sG7dOhYtWjTQz3g4XZLUpIceeoj58+ePRMABkjB//vwtOrJgxCVJzRqVgG+0pf89RlySpEYZcUmSGmXEJUmawJlnnsn73//+x5fPOOMMPvCBDwxxoicy4pIkTeD4449nxYoVAGzYsIELLriAY445ZshT/Ue+xUySpAksXLiQ+fPnc9VVV/G9732Pfffdl/nz5w97rP/AiEuStAknnHACy5cv58477+S4444b9jhP4OF0SZI2YdmyZVxyySWsXr2aQw89dNjjPIF74pIkbcLs2bM56KCDmDdvHrNmzRr2OE/Q2Z54kvOTfD/JdZt4PEk+kGRtkmuS7NfVLJIkTcWGDRtYtWoVxx9//LBHmVCXh9OXA4dN8vjhwOL+14nAOR3OIknSFrnhhhvYc889Ofjgg1m8ePGwx5lQZ4fTq+ryJAsn2eQI4K+qqoBVSeYleW5VfbermSRJGtSSJUu45ZZbhj3GpIZ5YdsuwG1jltf110mSpAEM88K2ie7yXhNumJxI75A7u+++e5czTc175g57gna85/5hT9AGf6cG5+/UYEbxd+rQC+GOGf4s8Z33ndnn69gw98TXAbuNWd4VuGOiDavqvKpaWlVLFyxYsFWGkyTpyW6YEb8YOLZ/lfoBwP2eD5ckaXCdHU5P8gng5cBOSdYB7wa2Baiqc4GVwKuAtcCDwJu7mkWSpFHU5dXpR2/m8QJO7ur1JUlPLQs/MOEZ2S30789x6x+8egaer1vedlWSpCm69dZbef7zn89b3vIWXvCCF3DIIYfw4x//mKuvvpoDDjiAvffem2XLlnHvvfd28vpGXJKkabj55ps5+eSTuf7665k3bx6f/OQnOfbYY3nf+97HNddcw1577cXZZ5/dyWsbcUmSpmHRokXss88+ALzoRS/iX//1X7nvvvt42cteBsAb3/hGLr/88k5e24hLkjQNT3/60x//ftasWdx3331b7bWNuCRJM2ju3Lk861nP4oorrgDgIx/5yON75TPNjyKVJGmGrVixgpNOOokHH3yQ5z3veXz4wx/u5HWMuCRpJNz69p2n/yRbeNvVhQsXct11//6J2+94xzse/37VqlXTn2czPJwuSVKjjLgkSY0y4pIkNcqIS5LUKCMuSVKjjLgkSY0y4pIkNcr3iUuSRsN5L5/Z53vP/TP7fB0w4pIkTdGZZ57JTjvtxKmnngrAGWecwXOe8xwefvhhLrzwQh5++GGWLVvG2WefzQMPPMCRRx7JunXreOyxxzjzzDM56qijpvX6Hk6XJGmKjj/+eFasWAHAhg0buOCCC3jOc57DzTffzNe+9jWuvvpq1qxZw+WXX84ll1zCzjvvzDe+8Q2uu+46DjvssGm/vhGXJGmKFi5cyPz587nqqqu47LLL2HfffVm9evXj3++3335885vf5Oabb2avvfbiC1/4AqeddhpXXHEFc+fOnfbrezhdkqRpOOGEE1i+fDl33nknxx13HF/84hc5/fTTeetb3/qEbdesWcPKlSs5/fTTOeSQQzjrrLOm9dpGXJKkaVi2bBlnnXUWjz76KB//+MfZZpttOPPMMznmmGOYM2cOt99+O9tuuy3r169nxx135A1veANz5sxh+fLl035tIy5J0jTMnj2bgw46iHnz5jFr1iwOOeQQbrzxRg488EAA5syZw0c/+lHWrl3LO9/5Tp72tKex7bbbcs4550z7tY24JGk0nPjl6T/HFn4UKfQuaFu1ahUXXXTR4+tOPfXUx69Y32iPPfbg0EMPnfaIY3lhmyRJU3TDDTew5557cvDBB7N48eKt/vruiUuSNEVLlizhlltuGdrruycuSVKjjLgkqVFF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\n",
      "text/plain": [
       "<Figure size 576x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(pd.crosstab(data['marital'],data['y']))\n",
    "\n",
    "marital=pd.crosstab(data['marital'],data['y'])\n",
    "marital.div(marital.sum(1).astype(float), axis=0).plot(kind=\"bar\", stacked=True, figsize=(8,8))\n",
    "plt.xlabel('marital')\n",
    "plt.ylabel('Percentage')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "73ab92b8",
   "metadata": {},
   "source": [
    "Here, We can see that singles and divorced clients are likely to more subscribed to insurance policy than married clients."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06ffbdba",
   "metadata": {},
   "source": [
    "Let's now look at how correlated our numerical variables are. We will see the correlation between each of these variables and the variable which have high negative or positive values are correlated. By this we can get an overview of the variables which might affect our target variable. We will convert our target variable into numeric values first."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2bccce35",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['y'].replace('no', 0,inplace=True)\n",
    "data['y'].replace('yes', 1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "42eca1ea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "corr = data.corr()\n",
    "mask = np.array(corr)\n",
    "mask[np.tril_indices_from(mask)] = False\n",
    "fig,ax= plt.subplots()\n",
    "fig.set_size_inches(20,10)\n",
    "sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap=\"YlGnBu\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c197bb8",
   "metadata": {},
   "source": [
    "We can infer that duration of the call is highly correlated with the target variable. This can be verified as well. As the duration of the call is more, there are higher chances that the client is showing interest in the term deposit and hence there are higher chances that the client will subscribe to term deposit."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a34bb4cb",
   "metadata": {},
   "source": [
    "Next we will look for any missing values in the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "67ef097c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age               0\n",
       "job               0\n",
       "marital           0\n",
       "education_qual    0\n",
       "call_type         0\n",
       "day               0\n",
       "mon               0\n",
       "dur               0\n",
       "num_calls         0\n",
       "prev_outcome      0\n",
       "y                 0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fce49e6a",
   "metadata": {},
   "source": [
    "There are no missing values in the dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4fcf1fd",
   "metadata": {},
   "source": [
    "As the sklearn models takes only numerical input, we will convert the categorical variables into numerical values using dummies. We'll apply dummies. We will also remove the target variable and keep it in a separate variable."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "891cf0cd",
   "metadata": {},
   "source": [
    "### Model Building"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "dbb4aa02",
   "metadata": {},
   "outputs": [],
   "source": [
    "target = data['y']\n",
    "data = data.drop('y',1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f739c5a8",
   "metadata": {},
   "source": [
    "Apply dummies on dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "70474274",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.get_dummies(data)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30cf2bf4",
   "metadata": {},
   "source": [
    "#### "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "61419288",
   "metadata": {},
   "source": [
    "We will split the train data into training and validation set so that we will be able to validate the results of our model on the validation set. We will keep 25% data as validation set and rest as the training set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "f84b4dcd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "5a38910e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size = 0.25, random_state = 42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1eb04c7a",
   "metadata": {},
   "source": [
    "Now our data is ready. For classification problem we gonna use Logistic Regression Model. Let's build the model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9362b6ac",
   "metadata": {},
   "source": [
    "#### \n",
    "### Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "455e3527",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "2849fa2a",
   "metadata": {},
   "outputs": [],
   "source": [
    "lreg = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "092dacf7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# fiting the model on our data.\n",
    "lreg.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "1e25c9a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "L_pred = lreg.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53502060",
   "metadata": {},
   "source": [
    "##### \n",
    "Now we will evaluate how accurate our predictions are. As the evaluation metric for this problem is AUC-ROC, let's calculate the accuracy on test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "ca61d9aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_auc_score, accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "09fef5cd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Score of Logistic Reg: 0.9\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy Score of Logistic Reg:\",round(accuracy_score(Y_test, L_pred),2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "53ba7733",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC Score of Logistic Reg: 0.66\n"
     ]
    }
   ],
   "source": [
    "L_AUC_ROC = roc_auc_score(Y_test, L_pred)\n",
    "print(\"ROC AUC Score of Logistic Reg:\",round((L_AUC_ROC),2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dfa0005c",
   "metadata": {},
   "source": [
    "Here for better ROC AUC SCORE we'll use Decision Tree Classifier Model. It do not have liniarity bound so let's what'll get if our data have non linearity?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "95434dd8",
   "metadata": {},
   "source": [
    " ### Decision Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "97d78635",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "6d63353d",
   "metadata": {},
   "outputs": [],
   "source": [
    "dtc = DecisionTreeClassifier(random_state=0, splitter='best')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d0acdbf2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(random_state=0)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dtc.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "04e8d6a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "D_pred = dtc.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "92844368",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Score of Decision Tree Cl: 0.9\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy Score of Decision Tree Cl:\",round(accuracy_score(Y_test, D_pred), 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "10b9c73b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC Score of Decision Tree Cl: 0.7\n"
     ]
    }
   ],
   "source": [
    "D_AUC_ROC = roc_auc_score(Y_test, D_pred)\n",
    "print(\"ROC AUC Score of Decision Tree Cl:\",round(D_AUC_ROC, 2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d4777b6c",
   "metadata": {},
   "source": [
    "#### \n",
    "### K Neighbors Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "d2233827",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "a2f84262",
   "metadata": {},
   "outputs": [],
   "source": [
    "KNC = KNeighborsClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "0ae9e435",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier()"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "KNC.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "30ccc50f",
   "metadata": {},
   "outputs": [],
   "source": [
    "K_pred = KNC.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "95867404",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Score of K Neighbors Cl: 0.88\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy Score of K Neighbors Cl:\",round(accuracy_score(Y_test, K_pred), 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "6c20f33f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC Score of K Neighbors Cl: 0.6\n"
     ]
    }
   ],
   "source": [
    "K_AUC_ROC = roc_auc_score(Y_test, K_pred)\n",
    "print(\"ROC AUC Score of K Neighbors Cl:\",round(K_AUC_ROC, 2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "96b14009",
   "metadata": {},
   "source": [
    "#### \n",
    "##### "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "708f5470",
   "metadata": {},
   "source": [
    "### "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23423fee",
   "metadata": {},
   "source": [
    "### Conclusion:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c1393b4",
   "metadata": {},
   "source": [
    "As our evalution metrix is **AUROC**. Here, we can see that our **Decision Tree Classifier** Model have **ROC AUC Score of 0.7** and as well as **Accuracy Score around 0.9**.\n",
    "\n",
    "So, We will use Desicion Tree Model and it'll predict output as much as 90% good. So it model's predicted output data will help our companies employ to direct target, agreeable or conversion clients more effectively and convert them to happy customer. So it'll save time of employ as well as company'.\n",
    "\n",
    "In the end this predictions will help company to run **time saver and cost effective** tele marketing campaigns with better customer conversion ratio.\n",
    "\n",
    "We can also design camaign were: employ will **connect to prospect clients first**, from the predicted output and successfully sell them term insurance."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}