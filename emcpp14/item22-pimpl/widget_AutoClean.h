
template<typename T>
class AutoClean
{
private:
	T *m_t;
public:
	AutoClean(T *pt) : m_t(pt) {}
	~AutoClean()
	{
		delete m_t;
	}

	operator bool() { return m_t != nullptr; }

	operator T*() { return m_t; }
	operator const T*() const { return m_t; }
};


class Widget
{
public:
	Widget();
	~Widget();

	// move-ctor/assign
	Widget(Widget&& rhs);
	Widget& operator=(Widget&& rhs);

	// copy-ctor/assign
	Widget(const Widget& rhs);
	Widget& operator=(const Widget& rhs);

#if 0
	static void test_incomplete_type()
	{
		AutoClean<Impl> aci(nullptr); // wacky-test
	}
#endif

private:
	struct Impl;
	
	AutoClean<Impl> pImpl;
	// std::unique_ptr<Impl> pImpl;
};
