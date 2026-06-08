
class CopyOnly {
public:
	CopyOnly() {
	}
	CopyOnly(int) {
	}
	CopyOnly(const CopyOnly&) = default;
	CopyOnly(CopyOnly&&) = delete; // explicitly deleted
};

CopyOnly ret() {
	return CopyOnly{}; // OK since C++17
}

int main()
{
	CopyOnly x = 42; // OK since C++17
	(void)x;
}
