{"aspt", [](c4::yml::ConstNodeRef options) -> method_factory_t<S> {
    return [](additional_options_t options, SpMMTask<S>& task) -> SpMMFunctor<S>* {
    return new SpMMASpT<S>(task);
    };
}}}